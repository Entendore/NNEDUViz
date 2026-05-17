"""
Educational Descriptions and Configuration
===========================================

All educational text displayed in the UI lives here.
"""

# ============================================================================
#  ARCHITECTURE DESCRIPTIONS (shown in info panel)
# ============================================================================

ARCHITECTURE_DESCRIPTIONS = {
    "mlp": """<h3 style='color:#89b4fa'>MLP (Multi-Layer Perceptron)</h3>
<p>The most fundamental neural network architecture. Every neuron in one
layer connects to every neuron in the next layer.</p>
<h4 style='color:#cba6f7'>How It Works:</h4>
<ul>
<li><b>Forward pass:</b> Input → Weighted sum → Activation → Output</li>
<li><b>Backpropagation:</b> Loss → Gradients → Weight updates</li>
<li><b>ReLU activation:</b> max(0, x) — simple but effective</li>
</ul>
<h4 style='color:#a6e3a1'>Key Property:</h4>
<p>A MLP with even one hidden layer can approximate <i>any</i> continuous
function (Universal Approximation Theorem). However, it may need many
neurons to learn simple patterns that specialized architectures handle naturally.</p>
<h4 style='color:#f9e2af'>Watch For:</h4>
<ul>
<li>Decision boundary becoming smoother with more epochs</li>
<li>Overfitting if network is too large for the dataset</li>
<li>How hidden layers transform the feature space (PCA plot)</li>
<li><b>Test vs Train gap:</b> Large gap = overfitting!</li>
</ul>""",

    "cnn": """<h3 style='color:#89b4fa'>CNN (1D Convolutional Neural Network)</h3>
<p>Designed to detect <b>local patterns</b> by sliding filters across input.</p>
<h4 style='color:#cba6f7'>How It Works:</h4>
<ul>
<li><b>Convolution:</b> A small filter slides over input, computing dot products</li>
<li><b>Pooling:</b> Reduces dimensionality, provides translation invariance</li>
<li><b>Weight sharing:</b> Same filter used everywhere → fewer parameters!</li>
</ul>
<h4 style='color:#a6e3a1'>Why CNN for Signals?</h4>
<p>The same wave pattern can appear at any position in a signal. CNN's
sliding window naturally handles this — unlike MLP which treats each
position independently.</p>
<h4 style='color:#f9e2af'>Parameter Efficiency:</h4>
<p>A CNN typically has <b>10-100x fewer parameters</b> than an MLP for the
same task because weights are shared across positions.</p>
<h4 style='color:#f38ba8'>Watch For:</h4>
<ul>
<li><b>Test accuracy lower than train?</b> The model may be memorizing specific wave positions</li>
<li>Confusion matrix shows which signal types are hardest to distinguish</li>
</ul>""",

    "rnn": """<h3 style='color:#89b4fa'>RNN (Recurrent Neural Network)</h3>
<p>Processes sequences step-by-step, maintaining a <b>hidden state</b>
that carries information from previous time steps.</p>
<h4 style='color:#cba6f7'>How It Works:</h4>
<ul>
<li>At each step: h<sub>t</sub> = tanh(W<sub>hh</sub>·h<sub>t-1</sub> + W<sub>xh</sub>·x<sub>t</sub>)</li>
<li>The same weights are reused at every time step</li>
<li>Hidden state h acts as "memory" of past inputs</li>
</ul>
<h4 style='color:#f38ba8'>The Vanishing Gradient Problem:</h4>
<p>During backpropagation through time, gradients are multiplied at each
step. For 100 steps with gradient factor 0.9: 0.9<sup>100</sup> ≈ 0.00003<br>
This means early time steps receive almost no learning signal!</p>
<h4 style='color:#f9e2af'>Watch For:</h4>
<ul>
<li>Compare R² with LSTM — LSTM should perform better</li>
<li>Gradient flow plot showing vanishing gradients</li>
<li><b>Test R² much lower than train?</b> Model memorized training sequences</li>
</ul>""",

    "lstm": """<h3 style='color:#89b4fa'>LSTM (Long Short-Term Memory)</h3>
<p>An enhanced RNN designed specifically to solve the vanishing gradient
problem through a gating mechanism.</p>
<h4 style='color:#cba6f7'>The Four Gates:</h4>
<table style='color:#cdd6f4; font-size:11px; border-collapse:collapse;'>
<tr><td style='color:#f38ba8'><b>Forget Gate (f)</b></td><td>What to discard from memory</td></tr>
<tr><td style='color:#a6e3a1'><b>Input Gate (i)</b></td><td>What new info to store</td></tr>
<tr><td style='color:#f9e2af'><b>Candidate (C̃)</b></td><td>Proposed new memory content</td></tr>
<tr><td style='color:#89b4fa'><b>Output Gate (o)</b></td><td>What to output now</td></tr>
</table>
<h4 style='color:#94e2d5'>The Cell State "Conveyor Belt":</h4>
<p>Unlike RNN's hidden state, LSTM's cell state C<sub>t</sub> has an
<b>additive</b> update path: C<sub>t</sub> = f<sub>t</sub> ⊙ C<sub>t-1</sub> + i<sub>t</sub> ⊙ C̃<sub>t</sub><br>
If forget gate f ≈ 1, gradients flow through unchanged — solving
vanishing gradients!</p>""",

    "transformer": """<h3 style='color:#89b4fa'>Transformer (Self-Attention)</h3>
<p>Processes all positions in <b>parallel</b> using attention, not recurrence.
This is the architecture behind GPT, BERT, and modern AI.</p>
<h4 style='color:#cba6f7'>Self-Attention Formula:</h4>
<p style='font-family:monospace; color:#f9e2af'>
Attention = softmax(QK<sup>T</sup> / √d<sub>k</sub>) V
</p>
<ul>
<li><b>Q (Query):</b> "What am I looking for?"</li>
<li><b>K (Key):</b> "What do I contain?"</li>
<li><b>V (Value):</b> "What information do I provide?"</li>
</ul>
<h4 style='color:#a6e3a1'>Advantages over RNN/LSTM:</h4>
<ul>
<li><b>Parallel:</b> All positions processed simultaneously (faster)</li>
<li><b>Long-range:</b> Direct attention between any two positions</li>
<li><b>Interpretable:</b> Attention weights show what the model focuses on</li>
</ul>
<h4 style='color:#f9e2af'>Attention Heatmap:</h4>
<p>Bright spots indicate strong attention. Watch for patterns — does the
model attend to adjacent positions? Periodic positions? This reveals
what relationships it learned.</p>""",

    "gan": """<h3 style='color:#89b4fa'>GAN (Generative Adversarial Network)</h3>
<p>Two networks competing in a game: Generator vs Discriminator.</p>
<h4 style='color:#cba6f7'>The Training Game:</h4>
<ul>
<li><b style='color:#a6e3a1'>Generator G:</b> Creates fake data from random noise.
Goal: fool the discriminator</li>
<li><b style='color:#f38ba8'>Discriminator D:</b> Scores real vs fake.
Goal: correctly identify fakes</li>
</ul>
<h4 style='color:#94e2d5'>Nash Equilibrium:</h4>
<p>Ideal outcome: G produces perfect fakes, D can't tell (50% accuracy).
In practice, reaching this balance is very challenging.</p>
<h4 style='color:#f38ba8'>Common Problems:</h4>
<ul>
<li><b>Mode collapse:</b> G only produces one type of output (missing modes)</li>
<li><b>D too strong:</b> G receives no useful gradients</li>
<li><b>Training instability:</b> G and D must improve together</li>
</ul>
<h4 style='color:#f9e2af'>Note on Test Sets:</h4>
<p>GANs don't use traditional train/test splits. The "test" is whether
generated samples match the <i>entire</i> target distribution, including
modes not well-represented in training data.</p>""",
}


# ============================================================================
#  DATASET DESCRIPTIONS
# ============================================================================

DATASET_DESCRIPTIONS = {
    "circles": """<h3 style='color:#89b4fa'>Concentric Circles</h3>
<p>Points on two concentric circles with added noise.</p>
<h4 style='color:#f9e2af'>Why This Dataset?</h4>
<p>A straight line cannot separate the circles — this is called
<b>linear inseparability</b>. The network must learn a curved
decision boundary.</p>
<p>This is the simplest non-trivial classification problem.</p>""",

    "spirals": """<h3 style='color:#89b4fa'>Interlocking Spirals</h3>
<p>Two spirals winding in opposite directions.</p>
<h4 style='color:#f38ba8'>Challenge Level: Very Hard</h4>
<p>The decision boundary must weave between tightly entangled classes.
This requires a network with sufficient capacity and many training epochs.</p>
<p>Good for testing: Can a deep network learn complex boundaries?</p>""",

    "xor": """<h3 style='color:#89b4fa'>XOR Pattern</h3>
<p>Four clusters at corners of a square with XOR labeling.</p>
<h4 style='color:#cba6f7'>Historical Significance:</h4>
<p>In 1969, Minsky & Papert proved that a <b>single neuron (perceptron)</b>
cannot solve XOR. This contributed to the first "AI Winter."</p>
<p>Solution: Add a hidden layer with at least 2 neurons. This allows
the network to learn two separating lines that combine to solve XOR.</p>""",

    "moons": """<h3 style='color:#89b4fa'>Half Moons</h3>
<p>Two interleaving half-circles with noise.</p>
<h4 style='color:#a6e3a1'>Moderate Difficulty</h4>
<p>A classic benchmark from scikit-learn. The boundary is curved but
not as complex as spirals. Most networks should learn this reasonably well.</p>""",

    "checkerboard": """<h3 style='color:#89b4fa'>Checkerboard</h3>
<p>Alternating class labels on a grid pattern.</p>
<h4 style='color:#f38ba8'>Challenge Level: Hard</h4>
<p>The repeating pattern creates <b>many local minima</b> — the optimizer
can get stuck in poor solutions. This tests the optimizer's ability to
escape local optima.</p>""",

    "signals_3class": """<h3 style='color:#89b4fa'>Signal Classification (3 Classes)</h3>
<p>Three types of 1D waveforms:</p>
<ul>
<li><b>Class 0:</b> sin(t) — fundamental frequency</li>
<li><b>Class 1:</b> sin(2t) — double frequency</li>
<li><b>Class 2:</b> sin(t)·cos(t) — amplitude modulated</li>
</ul>
<h4 style='color:#f9e2af'>Why CNN?</h4>
<p>CNN's sliding filters naturally detect these frequency patterns.
The same filter can find a wave pattern regardless of where it appears.</p>""",

    "signals_5class": """<h3 style='color:#89b4fa'>Signal Classification (5 Classes)</h3>
<p>Five distinct waveform types:</p>
<ul>
<li>Sine, Square wave, Sawtooth, Chirp, Damped oscillation</li>
</ul>
<p>More classes = harder discrimination. Watch the confusion matrix
to see which waveforms get confused.</p>""",

    "sine": """<h3 style='color:#89b4fa'>Sine Wave Prediction</h3>
<p>Given a sine wave sequence, predict the next value.</p>
<h4 style='color:#cba6f7'>The Challenge:</h4>
<p>Each sequence has <b>random phase and frequency</b>. The model must
learn the pattern of sine waves, not memorize specific sequences.</p>
<p>This tests whether the model learned the underlying function.</p>""",

    "multi_freq": """<h3 style='color:#89b4fa'>Multi-Frequency Prediction</h3>
<p>Predict next step of superimposed sine waves:</p>
<p style='font-family:monospace'>x<sub>t</sub> = 0.5·sin(f₁t) + 0.5·sin(f₂t)</p>
<h4 style='color:#f38ba8'>Harder than Single Frequency</h4>
<p>The model must learn to track multiple periodic components
simultaneously — more realistic for real-world signals.</p>""",

    "ring": """<h3 style='color:#89b4fa'>Ring Distribution</h3>
<p>Points arranged on a circle.</p>
<h4 style='color:#f9e2af'>GAN Challenge:</h4>
<p>The generator must learn a <b>1D manifold</b> (circle) embedded in 2D
space. It must produce points at the correct radius without spreading
out or collapsing to a point.</p>""",

    "spiral": """<h3 style='color:#89b4fa'>Spiral Distribution</h3>
<p>Points along an Archimedean spiral path.</p>
<h4 style='color:#f38ba8'>GAN Challenge:</h4>
<p>The spiral has curvature AND extends outward. The generator must
learn both the curved path and the increasing radius.</p>""",

    "gaussian_mix": """<h3 style='color:#89b4fa'>Gaussian Mixture (4 Modes)</h3>
<p>Four separate Gaussian clusters at corners of a square.</p>
<h4 style='color:#f38ba8'>GAN Challenge: Mode Collapse</h4>
<p>The biggest risk is <b>mode collapse</b> — the generator might only
produce points from 1-2 clusters while missing others. Watch if
generated points cover ALL four clusters!</p>""",

    "grid": """<h3 style='color:#89b4fa'>Grid Distribution</h3>
<p>Points arranged in a regular grid pattern.</p>
<h4 style='color:#f9e2af'>GAN Challenge:</h4>
<p>The discrete structure with gaps is challenging for GANs, which
naturally produce continuous distributions.</p>""",

    "figure8": """<h3 style='color:#89b4fa'>Figure-8 (Lemniscate)</h3>
<p>Points along a self-intersecting figure-8 curve.</p>
<h4 style='color:#f38ba8'>GAN Challenge:</h4>
<p>The self-intersection means the generator must learn that the same
region of space can have points from different parts of the curve.</p>""",
}


# ============================================================================
#  TRAINING TIPS (rotated during training)
# ============================================================================

TRAINING_TIPS = {
    "mlp": [
        "💡 Watch the decision boundary evolve from random to structured",
        "💡 Feature space (PCA) shows how network separates classes internally",
        "💡 Too many parameters → overfitting (test << train accuracy)",
        "💡 Test vs Train gap is the key overfitting indicator",
        "💡 Try different datasets to see which patterns MLP can learn",
        "💡 If test accuracy plateaus early, try adding dropout",
        "💡 Eigenvalue spectrum shows how weight structure evolves during training",
        "💡 Condition number rising? Optimization may become unstable",
    ],
    "cnn": [
        "💡 CNN has fewer parameters than MLP for the same task!",
        "💡 Convolutional filters learn to detect local wave patterns",
        "💡 Pooling provides translation invariance",
        "💡 Watch confusion matrix for per-class performance",
        "💡 Test accuracy much lower? Model may be memorizing wave positions",
        "💡 Larger test split = more reliable performance estimate",
        "💡 Singular values of conv filters show which frequencies are learned",
    ],
    "rnn": [
        "💡 RNN shares weights across time — very parameter efficient",
        "💡 Hidden state carries information from past to present",
        "💡 Compare R² with LSTM — LSTM should be better",
        "💡 Test R² much lower? Model memorized training sequences",
        "💡 Vanishing gradients limit RNN's long-term learning",
        "💡 Watch gradient flow plot for vanishing gradient warning signs",
        "💡 Recurrent weight eigenvalues reveal training stability",
    ],
    "lstm": [
        "💡 Cell state is the 'conveyor belt' — information flows unchanged",
        "💡 Gates decide what information to keep or discard",
        "💡 Compare performance with vanilla RNN",
        "💡 LSTM typically outperforms RNN on sequence tasks",
        "💡 Test set reveals if model learned the pattern or memorized",
        "💡 Watch for test R² dropping while train R² keeps rising",
        "💡 LSTM gate eigenvalues show how memory is managed",
    ],
    "transformer": [
        "💡 Attention heatmap shows which time steps the model focuses on",
        "💡 Parallel processing is faster than sequential RNN/LSTM",
        "💡 Positional encoding tells the model about sequence order",
        "💡 Multi-head attention learns different relationship patterns",
        "💡 Test set checks generalization to unseen sequences",
        "💡 Attention patterns on test data should look similar to train",
        "💡 Attention weight eigenvalues show learned focus patterns",
    ],
    "gan": [
        "💡 Generator and Discriminator must improve together",
        "💡 If D gets too strong, G can't learn (gradient vanishes)",
        "💡 If G gets too strong, D can't distinguish (mode collapse)",
        "💡 Watch for coverage: generated points should match real distribution",
        "💡 GANs don't use traditional test splits — evaluate on full distribution",
        "💡 Mode collapse is visible when generated points cluster in one area",
        "💡 Generator singular values show mode diversity",
    ],
}


# ============================================================================
#  EIGEN DESCRIPTIONS (educational content for eigen tab)
# ============================================================================

EIGEN_DESCRIPTIONS = {
    "spectrum": """<h3 style='color:#89b4fa'>Singular Value Spectrum</h3>
<p>Each weight matrix W is decomposed via SVD: W = U·Σ·V<sup>T</sup>.
The singular values σ₁ ≥ σ₂ ≥ ... describe the <b>importance</b> of
each direction in weight space.</p>
<h4 style='color:#cba6f7'>What to Look For:</h4>
<ul>
<li><b>Fast decay:</b> A few large singular values → low effective rank,
the network relies on few directions</li>
<li><b>Slow decay:</b> Many similar singular values → high effective rank,
the network uses many directions</li>
<li><b>Growing gap:</b> The largest singular values grow while small ones
shrink → the network is specializing</li>
</ul>
<h4 style='color:#a6e3a1'>Connection to Generalization:</h4>
<p>Networks with <b>flat singular value spectra</b> (many comparable
singular values) tend to generalize better than those with
<b>peaked spectra</b> (one dominant direction).</p>""",

    "condition": """<h3 style='color:#89b4fa'>Condition Number & Effective Rank</h3>
<p><b>Condition number</b> = σ<sub>max</sub> / σ<sub>min</sub></p>
<p>A high condition number means the weight matrix is <b>ill-conditioned</b>:
small changes in some input directions cause huge output changes,
while other directions are barely amplified.</p>
<h4 style='color:#f38ba8'>Why It Matters:</h4>
<ul>
<li><b>High condition number:</b> Gradients point in inconsistent
directions → slow or unstable training</li>
<li><b>Low condition number:</b> Well-conditioned → smoother optimization</li>
</ul>
<p><b>Effective rank</b> = number of singular values > 1% of σ<sub>max</sub></p>
<p>This measures how many <i>independent</i> directions the weight matrix
actually uses. A 64×64 matrix with effective rank 5 behaves more like
a rank-5 matrix.</p>""",

    "landscape": """<h3 style='color:#89b4fa'>Loss Landscape Slice</h3>
<p>Shows the loss when perturbing weights along the <b>top singular
vector</b> direction of the first weight layer.</p>
<h4 style='color:#cba6f7'>Reading the Plot:</h4>
<ul>
<li><b>Sharp narrow valley:</b> The loss increases rapidly away from
the minimum → <b>sharp minimum</b> → may generalize poorly</li>
<li><b>Wide flat valley:</b> The loss changes slowly → <b>flat minimum</b>
→ typically generalizes better</li>
<li><b>Asymmetric:</b> One side steeper than the other → the optimizer
may have difficulty approaching from certain directions</li>
</ul>
<h4 style='color:#a6e3a1'>Research Connection:</h4>
<p>Keskar et al. (2016) showed that flat minima generalize better than
sharp minima. The loss landscape curvature is a predictor of
generalization performance.</p>""",

    "eigenvectors": """<h3 style='color:#89b4fa'>Eigenvector Visualization</h3>
<p>The <b>right singular vectors</b> of a weight matrix represent the
input directions that the layer is most sensitive to.</p>
<h4 style='color:#cba6f7'>For MLP with 2D Input:</h4>
<p>The top singular vectors of the first layer can be plotted as
<b>arrows in input space</b>. These arrows show what directions
the network considers most important for transforming the input.</p>
<h4 style='color:#a6e3a1'>What to Watch:</h4>
<ul>
<li>Early training: arrows are random</li>
<li>Mid training: arrows start aligning with data structure</li>
<li>Late training: arrows align with the most discriminative directions</li>
</ul>
<h4 style='color:#f9e2af'>For Other Layers:</h4>
<p>Singular vectors are shown as heatmaps, revealing the internal
representation structure the network has learned.</p>""",
}


# ============================================================================
#  HYPERPARAMETER TOOLTIPS
# ============================================================================

HYPERPARAMETER_TOOLTIPS = {
    "learning_rate": """<b>Learning Rate</b><br><br>
The step size when updating weights. Too high causes instability;
too low causes slow training or getting stuck.<br><br>
<b>Typical values:</b> 0.001 - 0.01 for Adam<br>
<b>Symptoms of too high:</b> Loss increases or oscillates wildly<br>
<b>Symptoms of too low:</b> Loss decreases very slowly""",

    "weight_decay": """<b>Weight Decay (L2 Regularization)</b><br><br>
Penalizes large weights to prevent overfitting. Adds
λ·Σ(w²) to the loss.<br><br>
<b>Effect:</b> Encourages smaller, more distributed weights<br>
<b>When to use:</b> If test accuracy << training accuracy (overfitting)<br>
<b>Typical values:</b> 0.0001 - 0.01""",

    "dropout": """<b>Dropout Rate</b><br><br>
Randomly disables neurons during training (not inference).
This prevents co-adaptation of neurons.<br><br>
<b>Effect:</b> Network can't rely on any single neuron<br>
<b>When to use:</b> Large networks, overfitting<br>
<b>Typical values:</b> 0.2 - 0.5<br>
<b>Note:</b> 0 = disabled""",

    "epochs": """<b>Training Epochs</b><br><br>
One epoch = one full pass through the entire dataset.<br><br>
<b>Too few:</b> Underfitting — network hasn't learned<br>
<b>Too many:</b> Overfitting — network memorizes training data<br>
<b>Watch:</b> Loss plot to see when improvement plateaus<br>
<b>Tip:</b> If test metric starts decreasing while train increases, stop!""",

    "batch_size": """<b>Batch Size</b><br><br>
Number of samples processed before updating weights.<br><br>
<b>Small batch (4-32):</b> More noise in gradients, potentially better generalization<br>
<b>Large batch (64-256):</b> Smoother gradients, faster per-epoch but may generalize worse<br>
<b>Memory:</b> Larger batches require more GPU memory""",

    "test_split": """<b>Test Set Ratio</b><br><br>
Proportion of data held out for evaluation. The model NEVER sees
this data during training.<br><br>
<b>Why hold out data?</b><br>
Training accuracy can be misleadingly high — the model might just
be memorizing the training examples. Test accuracy shows how well
the model generalizes to <i>unseen</i> data.<br><br>
<b>Common values:</b> 0.2 (20% test, 80% train)<br>
<b>Too small:</b> Noisy/unreliable test metric<br>
<b>Too large:</b> Not enough training data<br><br>
<b>Overfitting detection:</b><br>
If train accuracy >> test accuracy, the model is overfitting!
Try: more dropout, weight decay, or a smaller network.""",

    "speed": """<b>Training Speed</b><br><br>
Controls the delay between epochs in milliseconds.
Lower values = faster training but less time to observe changes.<br><br>
<b>Slow (50-100ms):</b> Watch each epoch carefully<br>
<b>Fast (1-5ms):</b> Rush through epochs quickly<br>
<b>Default:</b> 10ms""",
}


# ============================================================================
#  OPTIMIZER TOOLTIPS
# ============================================================================

OPTIMIZER_TOOLTIPS = {
    "Adam": """<b>Adam (Adaptive Moment Estimation)</b><br><br>
Combines momentum (moving average of gradients) with RMSprop
(adaptive learning rates per parameter).<br><br>
<b>Pros:</b> Fast convergence, works well out-of-the-box<br>
<b>Cons:</b> May generalize slightly worse than SGD with tuning<br>
<b>Best for:</b> Most tasks, especially as a starting point""",

    "SGD": """<b>SGD (Stochastic Gradient Descent)</b><br><br>
The classic optimizer — updates weights using gradient of current batch.<br><br>
<b>Pros:</b> Often better generalization, simple<br>
<b>Cons:</b> Slow convergence, sensitive to learning rate<br>
<b>Tip:</b> Add momentum (0.9) for much better performance""",

    "RMSprop": """<b>RMSprop</b><br><br>
Adapts learning rate per parameter based on recent gradient magnitudes.
Good for non-stationary objectives.<br><br>
<b>Best for:</b> RNNs, online learning<br>
<b>Note:</b> Learning rate still needs careful tuning""",

    "AdamW": """<b>AdamW</b><br><br>
Adam with <b>decoupled weight decay</b>. Unlike regular Adam, weight
decay is applied separately from the gradient update.<br><br>
<b>Pros:</b> Better regularization than Adam, especially for transformers<br>
<b>Best for:</b> Transformer architectures, when using weight decay""",
}


# ============================================================================
#  ARCHITECTURE INFO
# ============================================================================

ARCHITECTURE_INFO = {
    "mlp": {
        "default_layers": [2, 64, 32, 2],
        "activation": "ReLU",
    },
    "cnn": {
        "architecture": "Conv1D(1→16) → Pool → Conv1D(16→32) → Pool → FC(64) → FC(3)",
    },
    "rnn": {"hidden_size": 32},
    "lstm": {"hidden_size": 32},
    "transformer": {"d_model": 32, "num_heads": 4, "num_layers": 2},
    "gan": {
        "latent_dim": 16,
        "generator": "FC(16→64) → FC(64→64) → FC(64→2)",
        "discriminator": "FC(2→64) → FC(64→64) → FC(64→1) → Sigmoid",
    },
}

GAN_MODES = ["ring", "spiral", "gaussian_mix", "grid", "figure8"]