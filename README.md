# zsmallnet

simple neural network designed for inlineable code optimizations

## Purpose

Small neural networks can execute in 20 nanoseconds, be deployed nearly anywhere, have no startup time, and solve way more problems than people give them credit for. This library is likely to be a good option when some combination of the following apply:

1. You need zero latency at startup (can't afford to wait to import tensorflow or pytorch)

2. You have tight constraints on throughput or battery life (need to process ten thousand touchpad events per second on a tiny fraction of a CPU core)

3. You need to build once and deploy anywhere (WASM, MacOS, Windows, Linux drivers, embedded devices, ...)

4. The function you're modeling isn't too complicated (scanning an image to find the corner coordinates of a curved page you want to turn into a pdf is fine, trying to be ChatGPT with a 1-hidden-layer inlined network probably isn't)

## Example Use Cases

The motivating use case for me writing this was a touchpad driver. I used assembly for the original, but this is the Zig code I would have wanted at the time. A network well under 2KB (including the optimized code) filtered out all the phantom touchpad movements while only tossing out 0.3% of the real movements (which had a net effect of making motions around 0.3% slower than they would have been). Neural networks are general function approximators, but other tasks that might be well suited include:

1. Adding an AI to the NPCs in games you're making

2. Detecting if a user is _actually_ navigating away from the page

3. Detecting a signal phrase to gate a larger computation (kind of like how "hey siri" or "ok google" might work under the hood)

4. Detecting the boundaries of a page in a mobile camera->pdf scanner app

5. Generating human-like mouse movements to wire to your captcha solver

And so on. Extremely tiny neural networks (7, 15, 31 hidden nodes) are surprisingly capable.

## Installation

Zig has a package manager!!! Do something like the following.

```zig
// build.zig.zon
.{
    .name = "foo",
    .version = "0.0.0",

    .dependencies = .{
        .zsmallnet = .{
            .name = "zsmallnet",
	    .url = "https://github.com/hmusgrave/zsmallnet/archive/refs/tags/0.0.1.tar.gz",
	    .hash = "122060be45ddd63e0ceb2bec7f42f8a002cdf27ea6ecbb376b2677ae0b04b1e2447e",
        },
    },
}
```

```zig
// build.zig
const zsmallnet_pkg = b.dependency("zsmallnet", .{
    .target = target,
    .optimize = optimize,
});
const zsmallnet_mod = zsmallnet_pkg.module("zsmallnet");
exe.addModule("zsmallnet", zsmallnet_mod);
unit_tests.addModule("zsmallnet", zsmallnet_mod);
```

## Examples

The public API consists of a couple builtin loss functions and a neural network type. The network has a well-defined (i.e., bitcastable) memory layout, so if you're not doing anything too fancy you can just bitcast to an array of `u8` (or mmap the thing) and use `@embedFile` to read the trained network back in at comptime (build.zig _could_ handle training and loading in the network artifact, but training the network can benefit from having a person in the loop).

```zig
const zsmallnet = @import("zsmallnet");

test "dimensions line up at comptime, and the network learns a little" {
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
    // 
    // as much as possible you want your hidden dimension
    // (the 7 below) to be 1 less than a power of 2, since
    // we add an extra dimension in the intermediate computations,
    // and that helps the compiler be able to generate
    // better instructions.
    var network: zsmallnet.MLPRegressor(F, I, 7, O) = undefined;
    network.init(rand, &x, &y);
    var last_err: F = undefined;

    // TRAINING
    // 
    // Often at this stage you would print/log/store errors and other
    // details, you might have an auto-updating graph, or you could be
    // able to tweak learning rates or other parameters on the fly.
    // You might want to keep the best network you've found so far in case
    // things fly off the rails. Etc.
    //
    // Almost none of that traditional deep learning stuff matters when
    // shoving a small neural network at a simple problem. If you get a bunch
    // of NaN out, decrease the learning rate. If the error barely changes,
    // increase it. If the error is decreasing but not enough, increase the
    // loop count. If you want better results but can't afford a bigger network
    // in your latency budget, re-initialize (WITH A NEW RANDOM STATE), and try
    // again, since smaller networks are less likely to find something
    // approximating the global optimum if the problem is hard.
    //
    // If you're messing with probabilities you'll want to use LogLoss. If you're
    // predicting discrete classes you'll want to replace those with probabilities
    // and then choose a method for threshholding (e.g., replace {dog|cat} with
    // {prob_dog, prob_cat} (maybe using something like 0.1,0.9 instead of 0,1 to
    // "smooth" the probabilities and help the network train faster), and then choose
    // to always pick the class with the highest probability).
    // 
    // The 3 different loops here are just examples to show common ways the API
    // might be used. Most applications would just use one.

    // Mean Squared Error
    // Note the "abs_reduction" which is like gradient descent
    // but moves the error a linear amount in linear loss regimes
    // (i.e., large gradients result in slower travels -- effectively a
    // hybrid Hessian approximation method)
    for (0..100) |_| {
        const grad = network.batch_grad(rand, 32, &x, &y, zsmallnet.MSE(F, O));
        network = network.abs_reduction(grad, 1e-3);
        last_err = grad.err;
    }

    // We can do perplexity (logloss/entropy) too. The choice of .auto clips
    // things as close to 0/1 as we can get away with without causing nan/inf
    // to crop up
    for (0..100) |_| {
        const grad = network.batch_grad(rand, 32, &x, &y, zsmallnet.LogLoss(F, O, .auto));
        network = network.gradient_descent(grad, 1e-3);
        last_err = grad.err;
    }

    // You can specify your favorite clipping epsilon instead. Also notice that we
    // have standard gradient descent here rather than the "abs_reduction".
    for (0..100) |_| {
        const grad = network.batch_grad(rand, 32, &x, &y, zsmallnet.LogLoss(F, O, .{ .float = 1e-50 }));
        network = network.gradient_descent(grad, 1e-3);
        last_err = grad.err;
    }

    // it's a tiny network, and we didn't train very long, so err isn't 0
    try std.testing.expect(last_err < 0.2);

    // don't forget to use the network for something
    _ = network.forward(.{2, 0.5});
}
```

## Status

Tested and working on my machine on zig-linux-x86_64-0.12.0-dev.86+197d9a9eb. Syntax changes (especially RTI for the casting builtins) will necessitate updates to work with much older compilers.
