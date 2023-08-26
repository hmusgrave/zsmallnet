const std = @import("std");
const Random = std.rand.Random;

fn Linear(comptime F: type, comptime I: usize, comptime O: usize) type {
    return extern struct {
        // standard "add a dimension" trick to lump the bias term
        // into a single matmul
        weights: [I + 1]@Vector(O, F),

        pub inline fn fill_normal(self: *@This(), rand: Random) void {
            // transforms unit gaussians to unit gaussians
            self.weights[I] = @splat(0);
            if (I < 1)
                return;
            const scalar: @Vector(O, F) = @splat(1 / @sqrt(@as(F, @floatFromInt(I))));
            for (self.weights[0..I]) |*w| {
                var arr_w: *[O]F = @ptrCast(w);
                for (arr_w) |*target|
                    target.* = rand.floatNorm(F);
                w.* *= scalar;
            }
        }

        // if (self) is initialized to transform unit gaussians to unit gaussians,
        // after mutating with self.match_input_moments(x) it will instead transform
        // gaussians with x's mean/variance (in each dimension) into unit gaussians
        //
        // Caution: No special care is given to suitably ignoring/handling near-zero-variance
        // inputs. We simply use @max(var, eps) to avoid divide-by-zero and do something reasonable,
        // but for particularly degenerate inputs (e.g., 99% of dimensions totally useless) that
        // might impact training.
        pub inline fn match_input_moments(self: *@This(), x: []const @Vector(I, F)) void {
            if (x.len < 1)
                return;

            const inv_count = 1 / @as(F, @floatFromInt(x.len));
            var mean: @Vector(I, F) = @splat(0);
            for (x) |v|
                mean += v;
            mean *= @splat(inv_count);

            var variance: @Vector(I, F) = @splat(0);
            for (x) |v|
                variance += (v - mean) * (v - mean);
            variance *= @splat(inv_count);
            variance = @max(variance, -variance);

            // TODO: throw an error instead?
            //
            // low variance input dimensions are treated roughly
            // as additional constant inputs, which screws with
            // the stats a bit
            const eps: @Vector(I, F) = @splat(1e-10);
            const denom = @max(variance, eps);

            const one: @Vector(I, F) = @splat(1);
            const reweight = @sqrt(one / denom);

            var bias: @Vector(O, F) = @splat(0);
            var reweight_arr: [I]F = reweight;
            var mean_arr: [I]F = mean;
            for (self.weights[0..I], reweight_arr, mean_arr) |*v, a, b| {
                bias += v.* * @as(@Vector(O, F), @splat(-a * b));
                v.* *= @splat(a);
            }
            self.weights[I] = bias;
        }

        // if (self) is initialized to transform unit gaussians to unit gaussians,
        // after mutating with self.match_output_moments(y) it will instead transform
        // unit gaussians to gaussians with y's mean/variance (in each dimension).
        pub inline fn match_output_moments(self: *@This(), y: []const @Vector(O, F)) void {
            if (y.len < 1)
                return;

            const inv_count = 1 / @as(F, @floatFromInt(y.len));
            var mean: @Vector(O, F) = @splat(0);
            for (y) |v|
                mean += v;
            mean *= @splat(inv_count);
            self.weights[I] = mean;

            var variance: @Vector(O, F) = @splat(0);
            for (y) |v|
                variance += (v - mean) * (v - mean);
            variance *= @splat(inv_count);
            variance = @max(variance, -variance);

            for (self.weights[0..I]) |*v|
                v.* *= @sqrt(variance);
        }

        // if (x @ self) produces unit gaussians then
        // elu(x @ self) has mean 0 and variance 1
        pub inline fn elu_whiten(self: *@This()) void {
            const sigma = 1.53996491;
            const mu = -4.86059367e-01;
            for (self.weights[0..I]) |*w| {
                w.* *= @splat(sigma);
            }

            self.weights[I] = @mulAdd(@Vector(O, F), self.weights[I], @splat(sigma), @splat(mu));
        }

        pub inline fn forward(self: @This(), _x: [I]F) [O]F {
            // x @ self.weights
            const x = _x ++ .{1};
            var rtn: @Vector(O, F) = @splat(0);
            for (x, self.weights) |el, v| {
                const w: @Vector(O, F) = @splat(el);
                rtn += v * w;
            }
            return rtn;
        }

        pub inline fn dX(self: @This(), dE: @Vector(O, F)) [I]F {
            // dE @ self.weights.transpose()
            var rtn = [_]F{0} ** I;
            for (&rtn, self.weights[0..I]) |*target, v|
                target.* = @reduce(.Add, v * dE);
            return rtn;
        }

        pub inline fn dM(_: @This(), _x: [I]F, dE: @Vector(O, F)) [I + 1]@Vector(O, F) {
            // x.transpose() @ dE
            const x = _x ++ .{1};
            var rtn: [I + 1]@Vector(O, F) = undefined;
            for (x, &rtn) |el, *target| {
                const w: @Vector(O, F) = @splat(el);
                target.* = w * dE;
            }
            return rtn;
        }
    };
}

fn Elu(comptime F: type, comptime n: usize) type {
    const V = @Vector(n, F);

    return struct {
        pub inline fn forward(x: V) V {
            const zero: V = @splat(0);
            const one: V = @splat(1);
            const lower = @exp(@select(F, x > zero, zero, x)) - one;
            const upper = @select(F, x > zero, x, zero);
            return lower + upper;
        }

        pub inline fn dX(dE: V) V {
            const zero: V = @splat(0);
            return @exp(@select(F, dE > zero, zero, dE));
        }
    };
}

pub fn MLPRegressor(comptime F: type, comptime I: usize, comptime H: usize, comptime O: usize) type {
    const E = Elu(F, H);

    return extern struct {
        layer0: Linear(F, I, H),
        layer1: Linear(F, H, O),

        pub inline fn init(self: *@This(), rand: Random, x: []const @Vector(I, F), y: []const @Vector(O, F)) void {
            self.layer0.fill_normal(rand);
            self.layer0.match_input_moments(x);
            self.layer0.elu_whiten();

            self.layer1.fill_normal(rand);
            self.layer1.match_output_moments(y);
        }

        pub inline fn forward(self: @This(), x: [I]F) [O]F {
            return self.layer1.forward(E.forward(self.layer0.forward(x)));
        }

        const Cache = struct {
            x0: [I]F,
            x1: [H]F,
            y: [O]F,
        };

        pub inline fn forward_for_dM(self: @This(), x: [I]F) Cache {
            var rtn: Cache = undefined;
            rtn.x0 = x;
            rtn.x1 = E.forward(self.layer0.forward(x));
            rtn.y = self.layer1.forward(rtn.x1);
            return rtn;
        }

        pub inline fn dX(self: @This(), dE: [O]F) [I]F {
            return self.layer0.dX(E.dX(self.layer1.dX(dE)));
        }

        pub inline fn dM(self: @This(), dE: [O]F, cache: Cache) @This() {
            var rtn: @This() = undefined;
            rtn.layer1.weights = self.layer1.dM(cache.x1, dE);
            rtn.layer0.weights = self.layer0.dM(cache.x0, E.dX(self.layer1.dX(dE)));
            return rtn;
        }

        pub inline fn add(self: @This(), other: @This()) @This() {
            var rtn: @This() = self;
            for (&rtn.layer0.weights, other.layer0.weights) |*target, w|
                target.* += w;
            for (&rtn.layer1.weights, other.layer1.weights) |*target, w|
                target.* += w;
            return rtn;
        }

        pub inline fn mul_const(self: @This(), c: F) @This() {
            var rtn: @This() = self;
            for (&rtn.layer0.weights) |*target|
                target.* *= @splat(c);
            for (&rtn.layer1.weights) |*target|
                target.* *= @splat(c);
            return rtn;
        }

        // TODO: naming, location
        const Self = @This();
        pub const EG = struct {
            err: F,
            grad: Self,
        };

        pub inline fn batch_grad(self: @This(), rand: Random, comptime n_batch: usize, x: []@Vector(I, F), y: []@Vector(O, F), comptime Loss: type) EG {
            var total_err: F = 0;
            var total_grad: @This() = self.mul_const(0);
            for (0..n_batch) |_| {
                const i = rand.intRangeLessThanBiased(usize, 0, x.len);
                const cache = self.forward_for_dM(x[i]);
                const loss = Loss.errgrad(cache.y, y[i]);
                total_err += loss.err;
                total_grad = total_grad.add(self.dM(loss.grad, cache));
            }
            if (n_batch < 1)
                return .{ .err = total_err, .grad = total_grad };
            const inv = 1 / @as(F, @floatFromInt(n_batch));
            return .{ .err = total_err * inv, .grad = total_grad.mul_const(inv) };
        }

        pub inline fn gradient_descent(self: @This(), grad: EG, eps: F) @This() {
            return self.add(grad.grad.mul_const(-eps));
        }

        pub inline fn abs_reduction(self: @This(), grad: EG, eps: F) @This() {
            var total0v: @Vector(H, F) = @splat(0);
            for (grad.grad.layer0.weights) |v|
                total0v += v * v;
            var total1v: @Vector(O, F) = @splat(0);
            for (grad.grad.layer1.weights) |v|
                total1v += v * v;
            const magnitude = @sqrt(@reduce(.Add, total0v) + @reduce(.Add, total1v));
            const scalar = if (magnitude < OneEps(F)) eps else eps / magnitude;
            return self.add(grad.grad.mul_const(-scalar));
        }
    };
}

pub inline fn ErrGrad(comptime F: type, comptime n: usize) type {
    return struct {
        err: F,
        grad: [n]F,
    };
}

pub fn MSE(comptime F: type, comptime n: usize) type {
    const V = @Vector(n, F);

    const fdim: F = @floatFromInt(n);
    const unsafe_inv_dim: F = if (n < 1) 1 else 1 / fdim;

    return struct {
        pub inline fn err(predicted: V, actual: V) F {
            const diff = predicted - actual;
            return @sqrt(@reduce(.Add, diff * diff) * unsafe_inv_dim);
        }

        pub inline fn errgrad(predicted: V, actual: V) ErrGrad(F, n) {
            const diff = predicted - actual;
            const s = @sqrt(@reduce(.Add, diff * diff));
            return .{
                .err = s * unsafe_inv_dim,
                .grad = diff * @as(V, @splat(unsafe_inv_dim / s)),
            };
        }
    };
}

const LogLossEps = union(enum) {
    auto: void,
    float: comptime_float,
};

fn OneEps(comptime F: type) F {
    // smallest positive eps such that 1-eps != 1
    var eps: F = 1;
    var n: usize = 0;
    while (1 != (1 - eps)) {
        eps /= 2;
        n += 1;
    }
    const nf: F = @floatFromInt(n);
    return @exp2(1 - nf);
}

// Log Loss is undefined for p=0 or p=1, so probabilities are clipped ε
// away from 0 and 1.
//
// A choice of .auto is fine for most applications, but experts might wish to
// experiment with specific values in .float
pub fn LogLoss(comptime F: type, comptime n: usize, comptime _eps: LogLossEps) type {
    const V = @Vector(n, F);

    const eps = comptime switch (_eps) {
        .auto => OneEps(F),
        .float => |f| f,
    };

    const zero: V = @splat(0);
    const one: V = @splat(1);
    const epsv: V = @splat(eps);

    return struct {
        pub inline fn err(_predicted: V, _actual: V) F {
            const actual = @max(@min(_actual, one), zero);
            const predicted = @max(@min(_predicted, one - epsv), epsv);
            return -@reduce(.Add, actual * @log2(predicted));
        }

        pub inline fn errgrad(_predicted: V, _actual: V) ErrGrad(F, n) {
            const inv_neg_log2: V = @splat(-1.0 / @log(@as(F, 2)));
            const actual = @max(@min(_actual, one), zero);
            const predicted = @max(@min(_predicted, one - epsv), epsv);
            return .{
                .err = -@reduce(.Add, actual * @log2(predicted)),
                .grad = actual * inv_neg_log2 / predicted,
            };
        }
    };
}

test "deterministic size and layout" {
    const M = MLPRegressor(f32, 2, 3, 4);
    const mat: M = undefined;

    // being bitcastable implies having a fixed memory layout
    const bytes: [@sizeOf(M)]u8 = @bitCast(mat);
    _ = bytes;
}

test "training dimensions match comptime and do something" {
    var prng = std.rand.DefaultPrng.init(42);
    const rand = prng.random();

    const F = f32;
    const I = 2;
    const O = 1;
    const n = 100;
    var x: [n]@Vector(I, F) = undefined;
    var y: [n]@Vector(O, F) = undefined;

    // trivial regression problem that also has outputs in (0,1) so that
    // we can test logloss in the same method
    for (&x, &y) |*u, *v| {
        const a = rand.floatNorm(F);
        const b = rand.floatNorm(F);
        u.* = .{ a, b };
        v.* = .{@max(a + b, -a - b) / 10.0};
    }

    // gotta have something to train
    var network: MLPRegressor(F, I, 8, O) = undefined;
    network.init(rand, &x, &y);
    var last_err: F = undefined;

    // Mean Squared Error
    // Note the "abs_reduction" which is like gradient descent
    // but moves the error a linear amount in linear loss regimes
    // (i.e., large gradients result in slower travels -- effectively a
    // hybrid Hessian approximation method)
    for (0..100) |_| {
        const grad = network.batch_grad(rand, 32, &x, &y, MSE(F, O));
        network = network.abs_reduction(grad, 1e-3);
        last_err = grad.err;
    }

    // We can do perplexity (logloss/entropy) too. The choice of .auto clips
    // things as close to 0/1 as we can get away with without causing nan/inf
    // to crop up
    for (0..100) |_| {
        const grad = network.batch_grad(rand, 32, &x, &y, LogLoss(F, O, .auto));
        network = network.gradient_descent(grad, 1e-3);
        last_err = grad.err;
    }

    // You can specify your favorite clipping epsilon instead. Also notice that we
    // have standard gradient descent here rather than the "abs_reduction".
    for (0..100) |_| {
        const grad = network.batch_grad(rand, 32, &x, &y, LogLoss(F, O, .{ .float = 1e-50 }));
        network = network.gradient_descent(grad, 1e-3);
        last_err = grad.err;
    }

    // it's a tiny network, and we didn't train very long, so err isn't 0
    try std.testing.expect(last_err < 0.2);
}
