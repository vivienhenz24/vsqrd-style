> Source: https://arxiv.org/abs/2105.06337

# Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech

Vadim Popov  Ivan Vovk  Vladimir Gogoryan  Tasnima Sadekova  Mikhail Kudinov 

###### Abstract

Recently, denoising diffusion probabilistic models and generative score matching have shown high potential in modelling complex data distributions while stochastic calculus has provided a unified point of view on these techniques allowing for flexible inference schemes. In this paper we introduce Grad-TTS, a novel text-to-speech model with score-based decoder producing mel-spectrograms by gradually transforming noise predicted by encoder and aligned with text input by means of Monotonic Alignment Search. The framework of stochastic differential equations helps us to generalize conventional diffusion probabilistic models to the case of reconstructing data from noise with different parameters and allows to make this reconstruction flexible by explicitly controlling trade-off between sound quality and inference speed. Subjective human evaluation shows that Grad-TTS is competitive with state-of-the-art text-to-speech approaches in terms of Mean Opinion Score. The code is publicly available at <https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS>.

Text-to-Speech, score matching, diffusion probabilistic modelling, SDE 

  

##  1 Introduction

Deep generative modelling proved to be effective in various machine learning fields, and speech synthesis is no exception. Modern text-to-speech (TTS) systems often consist of two parts designed as deep neural networks: the first part converts the input text into time-frequency domain acoustic features (feature generator), and the second one synthesizes raw waveform conditioned on these features (vocoder). Introduction of the conventional state-of-the-art autoregressive models such as Tacotron2 (Shen et al., 2018) used for feature generation and WaveNet (van den Oord et al., 2016) used as vocoder marked the beginning of the neural TTS era. Later, other popular generative modelling frameworks such as Generative Adversarial Networks (Goodfellow et al., 2014) and Normalizing Flows (Rezende & Mohamed, 2015) were used in the design of TTS engines for a parallel generation with comparable quality of the synthesized speech.

Since the publication of the WaveNet paper (2016), there have been various attempts to propose a parallel non-autoregressive vocoder, which could synthesize high-quality speech. Popular architectures based on Normalizing Flows like Parallel WaveNet (van den Oord et al., 2018) and WaveGlow (Prenger et al., 2019) managed to accelerate inference while keeping synthesis quality at a very high level but demonstrated fast synthesis on GPU devices only. Eventually, parallel GAN-based vocoders such as Parallel WaveGAN (Yamamoto et al., 2020), MelGAN (Kumar et al., 2019), and HiFi-GAN (Kong et al., 2020) greatly improved the performance of waveform generation on CPU devices. Furthermore, the latter model is reported to produce speech samples of state-of-the-art quality outperforming WaveNet.

Among feature generators, Tacotron2 (Shen et al., 2018) and Transformer-TTS (Li et al., 2019) enabled highly natural speech synthesis. Producing acoustic features frame by frame, they achieve almost perfect mel-spectrogram reconstruction from input text. Nonetheless, they often suffer from computational inefficiency and pronunciation issues coming from attention failures. Addressing these problems, such models as FastSpeech (Ren et al., 2019) and Parallel Tacotron (Elias et al., 2020) substantially improved inference speed and pronunciation robustness by utilizing non-autoregressive architectures and building hard monotonic alignments from estimated token lengths. However, in order to learn character duration, they still require pre-computed alignment from the teacher model. Finally, the recently proposed Non-Attentive Tacotron framework (Shen et al., 2020) managed to learn durations implicitly by employing the Variational Autoencoder concept.

Glow-TTS feature generator (Kim et al., 2020) based on Normalizing Flows can be considered as one of the most successful attempts to overcome pronunciation and computational latency issues typical for autoregressive solutions. Glow-TTS model made use of Monotonic Alignment Search algorithm (an adoption of Viterbi training (Rabiner, 1989) finding the most likely hidden alignment between two sequences) proposed to map the input text to mel-spectrograms efficiently. The alignment learned by Glow-TTS is intentionally designed to avoid some of the pronunciation problems models like Tacotron2 suffer from. Also, in order to enable parallel synthesis, Glow-TTS borrows encoder architecture from Transformer-TTS (Li et al., 2019) and decoder architecture from Glow (Kingma & Dhariwal, 2018). Thus, compared with Tacotron2, Glow-TTS achieves much faster inference making fewer alignment mistakes. Besides, in contrast to other parallel TTS solutions such as FastSpeech, Glow-TTS does not require an external aligner to obtain token duration information as Monotonic Alignment Search (MAS) operates in an unsupervised way.

Lately, another family of generative models called Diffusion Probabilistic Models (DPMs) (Sohl-Dickstein et al., 2015) has started to prove its capability to model complex data distributions such as images (Ho et al., 2020), shapes (Cai et al., 2020), graphs (Niu et al., 2020), handwriting (Luhman & Luhman, 2020). The basic idea behind DPMs is as follows: we build a forward diffusion process by iteratively destroying original data until we get some simple distribution (usually standard normal), and then we try to build a reverse diffusion parameterized with a neural network so that it follows the trajectories of the reverse-time forward diffusion. Stochastic calculus offers a continuous easy-to-use framework for training DPMs (Song et al., 2021) and, which is perhaps more important, provides a number of flexible inference schemes based on numerical differential equation solvers.

As far as text-to-speech applications are concerned, two vocoders representing the DPM family showed impressive results in raw waveform reconstruction: WaveGrad (Chen et al., 2021) and DiffWave (Kong et al., 2021) were shown to reproduce the fine-grained structure of human speech and match strong autoregressive baselines such as WaveNet in terms of synthesis quality while at the same time requiring much fewer sequential operations. However, despite such a success in neural vocoding, no feature generator based on diffusion probabilistic modelling is known so far.

This paper introduces Grad-TTS, an acoustic feature generator with a score-based decoder using recent diffusion probabilistic modelling insights. In Grad-TTS, MAS-aligned encoder outputs are passed to the decoder that transforms Gaussian noise parameterized by these outputs into a mel-spectrogram. To cope with the task of reconstructing data from Gaussian noise with varying parameters, we write down a generalized version of conventional forward and reverse diffusions. One of the remarkable features of our model is that it provides explicit control of the trade-off between output mel-spectrogram quality and inference speed. In particular, we find that Grad-TTS is capable of generating mel-spectrograms of high quality with only as few as ten iterations of reverse diffusion, which makes it possible to outperform Tacotron2 in terms of speed on GPU devices. Additionally, we show that it is possible to train Grad-TTS as an end-to-end TTS pipeline (i.e., vocoder and feature generator are combined in a single model) by replacing its output domain from mel-spectrogram to raw waveform.

##  2 Diffusion probabilistic modelling

Loosely speaking, a process of the diffusion type is a stochastic process that satisfies a stochastic differential equation (SDE)

| d​Xt=b​(Xt,t)​d​t+a​(Xt,t)​d​Wt,𝑑subscript𝑋𝑡𝑏subscript𝑋𝑡𝑡𝑑𝑡𝑎subscript𝑋𝑡𝑡𝑑subscript𝑊𝑡dX_{t}=b(X_{t},t)dt+a(X_{t},t)dW_{t}, |  | (1)  
---|---|---|---  
  
where Wtsubscript𝑊𝑡W_{t} is the standard Brownian motion, t∈[0,T]𝑡0𝑇t\in[0,T] for some finite time horizon T𝑇T, and coefficients b𝑏b and a𝑎a (called drift and diffusion correspondingly) satisfy certain measurability conditions. A rigorous definition of the diffusion type processes, as well as other notions from stochastic calculus we use in this section, can be found in (Liptser & Shiryaev, 1978).

It is easy to find such a stochastic process that terminal distribution L​a​w​(XT)𝐿𝑎𝑤subscript𝑋𝑇Law(X_{T}) converges to standard normal 𝒩​(0,I)𝒩0𝐼\mathcal{N}(0,I) when T→∞→𝑇T\to\infty for any initial data distribution L​a​w​(X0)𝐿𝑎𝑤subscript𝑋0Law(X_{0}) (I𝐼I is n×n𝑛𝑛n\times n identity matrix and n𝑛n is data dimensionality). In fact, there are lots of such processes as it follows from the formulae given later in this section. Any process of the diffusion type with such property is called forward diffusion and the goal of diffusion probabilistic modelling is to find a reverse diffusion such that its trajectories closely follow those of the forward diffusion but in reverse time order. This is, of course, a much harder task than making Gaussian noise out of data, but in many cases it still can be accomplished if we parameterize reverse diffusion with a proper neural network. In this case, generation boils down to sampling random noise from 𝒩​(0,I)𝒩0𝐼\mathcal{N}(0,I) and then just solving the SDE describing dynamics of the reverse diffusion with any numerical solver (usually a simple first-order Euler-Maruyama scheme (Kloeden & Platen, 1992) is used). If forward and reverse diffusion processes have close trajectories, then the distribution of resulting samples will be very close to that of the data L​a​w​(X0)𝐿𝑎𝑤subscript𝑋0Law(X_{0}). This approach to generative modelling is summarized in Figure 1.

  

Figure 1: Diffusion probabilistic modelling for mel-spectrograms.

Until recently, score-based and denoising diffusion probabilistic models were formalized in terms of Markov chains (Sohl-Dickstein et al., 2015; Song & Ermon, 2019; Ho et al., 2020; Song & Ermon, 2020). A unified approach introduced by Song et al. (2021) has demonstrated that these Markov chains actually approximated trajectories of stochastic processes satisfying certain SDEs. In our work, we follow this paper and define our DPM in terms of SDEs rather than Markov chains. As one can see later in Section 3, the task we are solving suggests generalizing DPMs described in (Song et al., 2021) in such a way that for infinite time horizon forward diffusion transforms any data distribution into 𝒩​(μ,Σ)𝒩𝜇Σ\mathcal{N}(\mu,\Sigma) instead of 𝒩​(0,I)𝒩0𝐼\mathcal{N}(0,I) for any given mean μ𝜇\mu and diagonal covariance matrix ΣΣ\Sigma. So, the rest of this section contains the detailed description of the generalized forward and reverse diffusions we utilize as well as the loss function we optimize to train the reverse diffusion. All corresponding derivations can be found in Appendix.

###  2.1 Forward diffusion

First, we need to define a forward diffusion process that transforms any data into Gaussian noise given infinite time horizon T𝑇T. If n𝑛n-dimensional stochastic process Xtsubscript𝑋𝑡X_{t} satisfies the following SDE:

| d​Xt=12​Σ−1​(μ−Xt)​βt​d​t+βt​d​Wt,t∈[0,T]formulae-sequence𝑑subscript𝑋𝑡12superscriptΣ1𝜇subscript𝑋𝑡subscript𝛽𝑡𝑑𝑡subscript𝛽𝑡𝑑subscript𝑊𝑡𝑡0𝑇dX_{t}=\frac{1}{2}\Sigma^{-1}(\mu-X_{t})\beta_{t}dt+\sqrt{\beta_{t}}dW_{t},\ \ \ t\in[0,T] |  | (2)  
---|---|---|---  
  
for non-negative function βtsubscript𝛽𝑡\beta_{t}, which we will refer to as noise schedule, vector μ𝜇\mu, and diagonal matrix ΣΣ\Sigma with positive elements, then its solution (if it exists) is given by

| Xt=(I−e−12​Σ−1​∫0tβs​𝑑s)​μ+e−12​Σ−1​∫0tβs​𝑑s​X0+∫0tβs​e−12​Σ−1​∫stβu​𝑑u​𝑑Ws.subscript𝑋𝑡𝐼superscript𝑒12superscriptΣ1superscriptsubscript0𝑡subscript𝛽𝑠differential-d𝑠𝜇superscript𝑒12superscriptΣ1superscriptsubscript0𝑡subscript𝛽𝑠differential-d𝑠subscript𝑋0superscriptsubscript0𝑡subscript𝛽𝑠superscript𝑒12superscriptΣ1superscriptsubscript𝑠𝑡subscript𝛽𝑢differential-d𝑢differential-dsubscript𝑊𝑠\begin{split}X_{t}&=\left(I-e^{-\frac{1}{2}\Sigma^{-1}\int_{0}^{t}\beta_{s}ds}\right)\mu+e^{-\frac{1}{2}\Sigma^{-1}\int_{0}^{t}\beta_{s}ds}X_{0}\\\ &+\int_{0}^{t}{\sqrt{\beta_{s}}e^{-\frac{1}{2}\Sigma^{-1}\int_{s}^{t}{\beta_{u}du}}dW_{s}}.\end{split} |  | (3)  
---|---|---|---  
  
Note that the exponential of a diagonal matrix is just an element-wise exponential. Let

| ρ​(X0,Σ,μ,t)=(I−e−12​Σ−1​∫0tβs​𝑑s)​μ+e−12​Σ−1​∫0tβs​𝑑s​X0𝜌subscript𝑋0Σ𝜇𝑡𝐼superscript𝑒12superscriptΣ1superscriptsubscript0𝑡subscript𝛽𝑠differential-d𝑠𝜇superscript𝑒12superscriptΣ1superscriptsubscript0𝑡subscript𝛽𝑠differential-d𝑠subscript𝑋0\begin{split}\rho(X_{0},\Sigma,\mu,t)&=\left(I-e^{-\frac{1}{2}\Sigma^{-1}\int_{0}^{t}\beta_{s}ds}\right)\mu\\\ &+e^{-\frac{1}{2}\Sigma^{-1}\int_{0}^{t}\beta_{s}ds}X_{0}\end{split} |  | (4)  
---|---|---|---  
  
and

| λ​(Σ,t)=Σ​(I−e−Σ−1​∫0tβs​𝑑s).𝜆Σ𝑡Σ𝐼superscript𝑒superscriptΣ1superscriptsubscript0𝑡subscript𝛽𝑠differential-d𝑠\lambda(\Sigma,t)=\Sigma\left(I-e^{-\Sigma^{-1}\int_{0}^{t}\beta_{s}ds}\right). |  | (5)  
---|---|---|---  
  
By properties of Itô’s integral conditional distribution of Xtsubscript𝑋𝑡X_{t} given X0subscript𝑋0X_{0} is Gaussian:

| L​a​w​(Xt|X0)=𝒩​(ρ​(X0,Σ,μ,t),λ​(Σ,t)).𝐿𝑎𝑤conditionalsubscript𝑋𝑡subscript𝑋0𝒩𝜌subscript𝑋0Σ𝜇𝑡𝜆Σ𝑡Law(X_{t}|X_{0})=\mathcal{N}(\rho(X_{0},\Sigma,\mu,t),\lambda(\Sigma,t)). |  | (6)  
---|---|---|---  
  
It means that if we consider infinite time horizon then for any noise schedule βtsubscript𝛽𝑡\beta_{t} such that limt→∞e−∫0tβs​𝑑s=0subscript→𝑡superscript𝑒superscriptsubscript0𝑡subscript𝛽𝑠differential-d𝑠0\lim_{t\to\infty}e^{-\int_{0}^{t}\beta_{s}ds}=0 we have

| Xt|X0→𝑑𝒩​(μ,Σ).𝑑→conditionalsubscript𝑋𝑡subscript𝑋0𝒩𝜇ΣX_{t}|X_{0}\xrightarrow{d}\mathcal{N}(\mu,\Sigma). |  | (7)  
---|---|---|---  
  
So, random variable Xtsubscript𝑋𝑡X_{t} converges in distribution to 𝒩​(μ,Σ)𝒩𝜇Σ\mathcal{N}(\mu,\Sigma) independently of X0subscript𝑋0X_{0}, and it is exactly the property we need: forward diffusion satisfying SDE (2) transforms any data distribution L​a​w​(X0)𝐿𝑎𝑤subscript𝑋0Law(X_{0}) into Gaussian noise 𝒩​(μ,Σ)𝒩𝜇Σ\mathcal{N}(\mu,\Sigma).

###  2.2 Reverse diffusion

While in earlier works on DPMs reverse diffusion was trained to approximate the trajectories of forward diffusion, Song et al. (2021) proposed to use the result by Anderson (1982), who derived an explicit formula for reverse-time dynamics of a wide class of stochastic processes of the diffusion type. In our case, this result leads to the following SDE for the reverse diffusion:

| d​Xt=(12​Σ−1​(μ−Xt)−∇log⁡pt​(Xt))​βt​d​t+βt​d​W~t,t∈[0,T],\begin{split}dX_{t}=&\left(\frac{1}{2}\Sigma^{-1}(\mu-X_{t})-\nabla\log{p_{t}(X_{t})}\right)\beta_{t}dt\\\ &+\sqrt{\beta_{t}}d\widetilde{W}_{t},\qquad\qquad\qquad\qquad t\in[0,T],\end{split} |  | (8)  
---|---|---|---  
  
where W~tsubscript~𝑊𝑡\widetilde{W}_{t} is the reverse-time Brownian motion and ptsubscript𝑝𝑡p_{t} is the probability density function of random variable Xtsubscript𝑋𝑡X_{t}. This SDE is to be solved backwards starting from terminal condition XTsubscript𝑋𝑇X_{T}.

Moreover, Song et al. (2021) have shown that instead of SDE (8), we can consider an ordinary differential equation

| d​Xt=12​(Σ−1​(μ−Xt)−∇log⁡pt​(Xt))​βt​d​t.𝑑subscript𝑋𝑡12superscriptΣ1𝜇subscript𝑋𝑡∇subscript𝑝𝑡subscript𝑋𝑡subscript𝛽𝑡𝑑𝑡dX_{t}=\frac{1}{2}\left(\Sigma^{-1}(\mu-X_{t})-\nabla\log{p_{t}(X_{t})}\right)\beta_{t}dt. |  | (9)  
---|---|---|---  
  
Forward Kolmogorov equations corresponding to (2) and (9) are identical, which means that the evolution of probability density functions of stochastic processes given by (2) and (9) is the same.

Thus, if we have a neural network sθ​(Xt,t)subscript𝑠𝜃subscript𝑋𝑡𝑡s_{\theta}(X_{t},t) that estimates the gradient of the log-density of noisy data ∇log⁡pt​(Xt)∇subscript𝑝𝑡subscript𝑋𝑡\nabla\log{p_{t}(X_{t})}, then we can model data distribution L​a​w​(X0)𝐿𝑎𝑤subscript𝑋0Law(X_{0}) by sampling XTsubscript𝑋𝑇X_{T} from 𝒩​(μ,Σ)𝒩𝜇Σ\mathcal{N}(\mu,\Sigma) and numerically solving either (8) or (9) backwards in time.

###  2.3 Loss function

Estimating gradients of log-density of noisy data Xtsubscript𝑋𝑡X_{t} is often referred to as score matching, and in recent papers (Song & Ermon, 2019, 2020) L2subscript𝐿2L_{2} loss was used to approximate these gradients with a neural network. So, in our paper, we use the same type of loss.

Figure 2: Grad-TTS inference scheme.

Due to the formula (6), we can sample noisy data Xtsubscript𝑋𝑡X_{t} given only initial data X0subscript𝑋0X_{0} without sampling intermediate values {Xs}s<tsubscriptsubscript𝑋𝑠𝑠𝑡\\{X_{s}\\}_{s<t}. Moreover, L​a​w​(Xt|X0)𝐿𝑎𝑤conditionalsubscript𝑋𝑡subscript𝑋0Law(X_{t}|X_{0}) is Gaussian, which means that its log-density has a very simple closed form. If we sample ϵtsubscriptitalic-ϵ𝑡\epsilon_{t} from 𝒩​(0,λ​(Σ,t))𝒩0𝜆Σ𝑡\mathcal{N}(0,\lambda(\Sigma,t)) and then put

| Xt=ρ​(X0,Σ,μ,t)+ϵtsubscript𝑋𝑡𝜌subscript𝑋0Σ𝜇𝑡subscriptitalic-ϵ𝑡X_{t}=\rho(X_{0},\Sigma,\mu,t)+\epsilon_{t} |  | (10)  
---|---|---|---  
  
in accordance with (6), then the gradient of log-density of noisy data in this point Xtsubscript𝑋𝑡X_{t} is given by

| ∇log⁡p0​t​(Xt|X0)=−λ​(Σ,t)−1​ϵt,∇subscript𝑝0𝑡conditionalsubscript𝑋𝑡subscript𝑋0𝜆superscriptΣ𝑡1subscriptitalic-ϵ𝑡\nabla\log{p_{0t}(X_{t}|X_{0})}=-\lambda(\Sigma,t)^{-1}\epsilon_{t}, |  | (11)  
---|---|---|---  
  
where p0​t(⋅|X0)p_{0t}(\cdot|X_{0}) is the probability density function of the conditional distribution (6). Thus, loss function corresponding to estimating the gradient of log-density of data X0subscript𝑋0X_{0} corrupted with noise accumulated by time t𝑡t is

| ℒt​(X0)=𝔼ϵt​[‖sθ​(Xt,t)+λ​(Σ,t)−1​ϵt‖22],subscriptℒ𝑡subscript𝑋0subscript𝔼subscriptitalic-ϵ𝑡delimited-[]superscriptsubscriptnormsubscript𝑠𝜃subscript𝑋𝑡𝑡𝜆superscriptΣ𝑡1subscriptitalic-ϵ𝑡22\mathcal{L}_{t}(X_{0})=\mathbb{E}_{\epsilon_{t}}\left[\left\|{s_{\theta}(X_{t},t)+\lambda(\Sigma,t)^{-1}\epsilon_{t}}\right\|_{2}^{2}\right], |  | (12)  
---|---|---|---  
  
where ϵtsubscriptitalic-ϵ𝑡\epsilon_{t} is sampled from 𝒩​(0,λ​(Σ,t))𝒩0𝜆Σ𝑡\mathcal{N}(0,\lambda(\Sigma,t)) and Xtsubscript𝑋𝑡X_{t} is calculated by formula (10).

##  3 Grad-TTS

The acoustic feature generator we propose consists of three modules: encoder, duration predictor, and decoder. In this section, we will describe their architectures as well as training and inference procedures. The general approach is illustrated in Figure 2. Grad-TTS has very much in common with Glow-TTS (Kim et al., 2020), a feature generator based on Normalizing Flows. The key difference lies in the principles the decoder relies on.

###  3.1 Inference

An input text sequence x1:Lsubscript𝑥:1𝐿x_{1:L} of length L𝐿L typically consists of characters or phonemes, and we aim at generating mel-spectrogram y1:Fsubscript𝑦:1𝐹y_{1:F} where F𝐹F is the number of acoustic frames. In Grad-TTS, the encoder converts an input text sequence x1:Lsubscript𝑥:1𝐿x_{1:L} into a sequence of features μ~1:Lsubscript~𝜇:1𝐿\tilde{\mu}_{1:L} used by the duration predictor to produce hard monotonic alignment A𝐴A between encoded text sequence μ~1:Lsubscript~𝜇:1𝐿\tilde{\mu}_{1:L} and frame-wise features μ1:Fsubscript𝜇:1𝐹\mu_{1:F}. The function A𝐴A is a monotonic surjective mapping between [1,F]∩ℕ1𝐹ℕ[1,F]\cap\mathbb{N} and [1,L]∩ℕ1𝐿ℕ[1,L]\cap\mathbb{N}, and we put μj=μ~A​(j)subscript𝜇𝑗subscript~𝜇𝐴𝑗\mu_{j}=\tilde{\mu}_{A(j)} for any integer j∈[1,F]𝑗1𝐹j\in[1,F]. Informally speaking, the duration predictor tells us how many frames each element of text input lasts. Monotonicity and surjectiveness of A𝐴A guarantee that the text is pronounced in the correct order without skipping any text input. As in all TTS models with duration predictor, it is possible to control synthesized speech tempo by multiplying predicted durations by some factor.

The output sequence μ=μ1:F𝜇subscript𝜇:1𝐹\mu=\mu_{1:F} is then passed to the decoder, which is a Diffusion Probabilistic Model. A neural network sθ​(Xt,μ,t)subscript𝑠𝜃subscript𝑋𝑡𝜇𝑡s_{\theta}(X_{t},\mu,t) with parameters θ𝜃\theta defines an ordinary differential equation (ODE)

| d​Xt=12​(μ−Xt−sθ​(Xt,μ,t))​βt​d​t,𝑑subscript𝑋𝑡12𝜇subscript𝑋𝑡subscript𝑠𝜃subscript𝑋𝑡𝜇𝑡subscript𝛽𝑡𝑑𝑡dX_{t}=\frac{1}{2}(\mu-X_{t}-s_{\theta}(X_{t},\mu,t))\beta_{t}dt, |  | (13)  
---|---|---|---  
  
which is solved backwards in time using the first-order Euler scheme. The sequence μ𝜇\mu is also used to define the terminal condition XT∼𝒩​(μ,I)similar-tosubscript𝑋𝑇𝒩𝜇𝐼X_{T}\sim\mathcal{N}(\mu,I). Noise schedule βtsubscript𝛽𝑡\beta_{t} and time horizon T𝑇T are some pre-defined hyperparameters whose choice mostly depends on the data, while step size hℎh in the Euler scheme is a hyperparameter that can be chosen after Grad-TTS is trained. It expresses the trade-off between the quality of output mel-spectrograms and inference speed.

Reverse diffusion in Grad-TTS evolves according to equation (13) for the following reasons:

  * •

We obtained better results in practice when using dynamics (9) instead of (8): for small values of step size hℎh, they performed equally well, while for larger values the former led to much better sounding results.

  * •

We chose Σ=IΣ𝐼\Sigma=I to simplify the whole feature generation pipeline.

  * •

We used μ𝜇\mu as an additional input to the neural network sθ​(Xt,μ,t)subscript𝑠𝜃subscript𝑋𝑡𝜇𝑡s_{\theta}(X_{t},\mu,t). It follows from (11) that the neural network sθsubscript𝑠𝜃s_{\theta} essentially tries to predict Gaussian noise added to data X0subscript𝑋0X_{0} observing only noisy data Xtsubscript𝑋𝑡X_{t}. So, if for every time t𝑡t we supply sθsubscript𝑠𝜃s_{\theta} with an additional knowledge of how the limiting noise limT→∞L​a​w​(XT|X0)subscript→𝑇𝐿𝑎𝑤conditionalsubscript𝑋𝑇subscript𝑋0\lim_{T\to\infty}Law(X_{T}|X_{0}) looks like (note that it is different for different text input), then this network can make more accurate predictions of noise at time t∈[0,T]𝑡0𝑇t\in[0,T].

We also found it beneficial for the model performance to introduce a temperature hyperparameter τ𝜏\tau and to sample terminal condition XTsubscript𝑋𝑇X_{T} from 𝒩​(μ,τ−1​I)𝒩𝜇superscript𝜏1𝐼\mathcal{N}(\mu,\tau^{-1}I) instead of 𝒩​(μ,I)𝒩𝜇𝐼\mathcal{N}(\mu,I). Tuning τ𝜏\tau can help to keep the quality of output mel-spectrograms at the same level when using larger values of step size hℎh.

###  3.2 Training

One of Grad-TTS training objectives is to minimize the distance between aligned encoder output μ𝜇\mu and target mel-spectrogram y𝑦y because the inference scheme that has just been described suggests to start decoding from random noise 𝒩​(μ,I)𝒩𝜇𝐼\mathcal{N}(\mu,I). Intuitively, it is clear that decoding is easier if we start from noise, which is already close to the target y𝑦y in some sense.

If aligned encoder output μ𝜇\mu is considered to parameterize an input noise the decoder starts from, it is natural to regard encoder output μ~~𝜇\tilde{\mu} as a normal distribution 𝒩​(μ~,I)𝒩~𝜇𝐼\mathcal{N}(\tilde{\mu},I), which leads to a negative log-likelihood encoder loss:

| ℒe​n​c=−∑j=1Flog⁡φ​(yj;μ~A​(j),I),subscriptℒ𝑒𝑛𝑐superscriptsubscript𝑗1𝐹𝜑subscript𝑦𝑗subscript~𝜇𝐴𝑗𝐼\mathcal{L}_{enc}=-\sum_{j=1}^{F}{\log{\varphi(y_{j};\tilde{\mu}_{A(j)},I)}}, |  | (14)  
---|---|---|---  
  
where φ​(⋅;μ~i,I)𝜑⋅subscript~𝜇𝑖𝐼\varphi(\cdot;\tilde{\mu}_{i},I) is a probability density function of 𝒩​(μ~i,I)𝒩subscript~𝜇𝑖𝐼\mathcal{N}(\tilde{\mu}_{i},I). Although other types of losses are also possible, we have chosen ℒe​n​csubscriptℒ𝑒𝑛𝑐\mathcal{L}_{enc} (which actually reduces to Mean Square Error criterion) because of this probabilistic interpretation. In principle, it is even possible to train Grad-TTS without any encoder loss at all and let the diffusion loss described further do all the job of generating realistic mel-spectrograms, but in practice we observed that in the absence of ℒe​n​csubscriptℒ𝑒𝑛𝑐\mathcal{L}_{enc} Grad-TTS failed to learn alignment.

The encoder loss ℒe​n​csubscriptℒ𝑒𝑛𝑐\mathcal{L}_{enc} has to be optimized with respect to both encoder parameters and alignment function A𝐴A. Since it is hard to do a joint optimization, we apply an iterative approach proposed by Kim et al. (2020). Each iteration of optimization consists of two steps: (i) searching for an optimal alignment A∗superscript𝐴A^{*} given fixed encoder parameters; (ii) fixing this alignment A∗superscript𝐴A^{*} and taking one step of stochastic gradient descent to optimize loss function with respect to encoder parameters. We use Monotonic Alignment Search at the first step of this approach. MAS utilizes the concept of dynamic programming to find an optimal (from the point of view of loss function ℒe​n​csubscriptℒ𝑒𝑛𝑐\mathcal{L}_{enc}) monotonic surjective alignment. This algorithm is described in detail in (Kim et al., 2020).

To estimate the optimal alignment A∗superscript𝐴A^{*} at inference, Grad-TTS employs the duration predictor network. As in (Kim et al., 2020), we train the duration predictor D​P𝐷𝑃DP with Mean Square Error (MSE) criterion in logarithmic domain:

| di=log∑j=1F𝕀{A∗​(j)=i},i=1,..,L,ℒd​p=M​S​E​(D​P​(s​g​[μ~]),d),\begin{split}d_{i}=&\log{\sum_{j=1}^{F}{\mathbb{I}_{\\{A^{*}(j)=i\\}}}},\ \ \ i=1,..,L,\\\ &\mathcal{L}_{dp}=MSE(DP(sg[\tilde{\mu}]),d),\end{split} |  | (15)  
---|---|---|---  
  
where 𝕀𝕀\mathbb{I} is an indicator function, μ~=μ~1:L~𝜇subscript~𝜇:1𝐿\tilde{\mu}=\tilde{\mu}_{1:L}, d=d1:L𝑑subscript𝑑:1𝐿d=d_{1:L} and stop gradient operator s​g​[⋅]𝑠𝑔delimited-[]⋅sg[\cdot] is applied to the inputs of the duration predictor to prevent ℒd​psubscriptℒ𝑑𝑝\mathcal{L}_{dp} from affecting encoder parameters.

As for the loss related to the DPM, it is calculated using formulae from Section 2. As already mentioned, we put Σ=IΣ𝐼\Sigma=I, so the distribution of noisy data (6) simplifies, and its covariance matrix becomes just an identity matrix I𝐼I multiplied by a scalar

| λt=1−e−∫0tβs​𝑑s.subscript𝜆𝑡1superscript𝑒superscriptsubscript0𝑡subscript𝛽𝑠differential-d𝑠\lambda_{t}=1-e^{-\int_{0}^{t}{\beta_{s}ds}}. |  | (16)  
---|---|---|---  
  
The overall diffusion loss function ℒd​i​f​fsubscriptℒ𝑑𝑖𝑓𝑓\mathcal{L}_{diff} is the expectation of weighted losses associated with estimating gradients of log-density of noisy data at different times t∈[0,T]𝑡0𝑇t\in[0,T]:

| ℒd​i​f​f=𝔼X0,t​[λt​𝔼ξt​[‖sθ​(Xt,μ,t)+ξtλt‖22]],subscriptℒ𝑑𝑖𝑓𝑓subscript𝔼subscript𝑋0𝑡delimited-[]subscript𝜆𝑡subscript𝔼subscript𝜉𝑡delimited-[]superscriptsubscriptnormsubscript𝑠𝜃subscript𝑋𝑡𝜇𝑡subscript𝜉𝑡subscript𝜆𝑡22\mathcal{L}_{diff}=\mathbb{E}_{X_{0},t}\left[\lambda_{t}\mathbb{E}_{\xi_{t}}\left[\left\|s_{\theta}(X_{t},\mu,t)+\frac{\xi_{t}}{\sqrt{\lambda_{t}}}\right\|_{2}^{2}\right]\right], |  | (17)  
---|---|---|---  
  
where X0subscript𝑋0X_{0} stands for target mel-spectrogram y𝑦y sampled from training data, t𝑡t is sampled from uniform distribution on [0,T]0𝑇[0,T], ξtsubscript𝜉𝑡\xi_{t} – from 𝒩​(0,I)𝒩0𝐼\mathcal{N}(0,I) and the formula

| Xt=ρ​(X0,I,μ,t)+λt​ξtsubscript𝑋𝑡𝜌subscript𝑋0𝐼𝜇𝑡subscript𝜆𝑡subscript𝜉𝑡X_{t}=\rho(X_{0},I,\mu,t)+\sqrt{\lambda_{t}}\xi_{t} |  | (18)  
---|---|---|---  
  
is used to get noisy data Xtsubscript𝑋𝑡X_{t} according to the distribution (6). The above formulae (17) and (18) follow from (12) and (10) by substitution ϵt=λt​ξtsubscriptitalic-ϵ𝑡subscript𝜆𝑡subscript𝜉𝑡\epsilon_{t}=\sqrt{\lambda_{t}}\xi_{t}. We use losses (12) with weights λtsubscript𝜆𝑡\lambda_{t} according to the common heuristics that these weights should be proportional to 1/𝔼[∥∇logp0​t(Xt|X0)∥22]1/\mathbb{E}\left[\left\|\nabla\log{p_{0t}(X_{t}|X_{0})}\right\|_{2}^{2}\right].

To sum it up, the training procedure consists of the following steps:

  * •

Fix the encoder, duration predictor, and decoder parameters and run MAS algorithm to find the alignment A∗superscript𝐴A^{*} that minimizes ℒe​n​csubscriptℒ𝑒𝑛𝑐\mathcal{L}_{enc}.

  * •

Fix the alignment A∗superscript𝐴A^{*} and minimize ℒe​n​c+ℒd​p+ℒd​i​f​fsubscriptℒ𝑒𝑛𝑐subscriptℒ𝑑𝑝subscriptℒ𝑑𝑖𝑓𝑓\mathcal{L}_{enc}+\mathcal{L}_{dp}+\mathcal{L}_{diff} with respect to encoder, duration predictor, and decoder parameters.

  * •

Repeat the first two steps till convergence.

###  3.3 Model architecture

As for the encoder and duration predictor, we use exactly the same architectures as in Glow-TTS, which in its turn borrows the structure of these modules from Transformer-TTS (Li et al., 2019) and FastSpeech (Ren et al., 2019) correspondingly. The duration predictor consists of two convolutional layers followed by a projection layer that predicts the logarithm of duration. The encoder is composed of a pre-net, 666 Transformer blocks with multi-head self-attention, and the final linear projection layer. The pre-net consists of 333 layers of convolutions followed by a fully-connected layer.

The decoder network sθsubscript𝑠𝜃s_{\theta} has the same U-Net architecture (Ronneberger et al., 2015) used by Ho et al. (2020) to generate 32×32323232\times 32 images, except that we use twice fewer channels and three feature map resolutions instead of four to reduce model size. In our experiments we use 808080-dimensional mel-spectrograms, so sθsubscript𝑠𝜃s_{\theta} operates on resolutions 80×F80𝐹80\times F, 40×F/240𝐹240\times F/2 and 20×F/420𝐹420\times F/4. We zero-pad mel-spectrograms if the number of frames F𝐹F is not a multiple of 444. Aligned encoder output μ𝜇\mu is concatenated with U-Net input Xtsubscript𝑋𝑡X_{t} as an additional channel.

##  4 Experiments

LJSpeech dataset (Ito, 2017) containing approximately 242424 hours of English female voice recordings sampled at 22.0522.0522.05kHz was used to train the Grad-TTS model. The test set contained around 500500500 short audio recordings (duration less than 101010 seconds each). The input text was phonemized before passing to the encoder; as for the output acoustic features, we used conventional 808080-dimensional mel-spectrograms. We tried training both on original and normalized mel-spectrograms and found that the former performed better. Grad-TTS was trained for 1.7​m1.7𝑚1.7m iterations on a single GPU (NVIDIA RTX 208020802080 Ti with 111111GB memory) with mini-batch size 161616. We chose Adam optimizer and set the learning rate to 0.00010.00010.0001.

Figure 3: Diffusion loss at training.

We would like to mention several important things about Grad-TTS training:

  * •

We chose T=1𝑇1T=1, βt=β0+(β1−β0)​tsubscript𝛽𝑡subscript𝛽0subscript𝛽1subscript𝛽0𝑡\beta_{t}=\beta_{0}+(\beta_{1}-\beta_{0})t, β0=0.05subscript𝛽00.05\beta_{0}=0.05 and β1=20subscript𝛽120\beta_{1}=20.

  * •

As in (Bińkowski et al., 2020; Donahue et al., 2021), we use random mel-spectrogram segments of fixed length (222 seconds in our case) as training targets y𝑦y to allow for memory-efficient training. However, MAS and the duration predictor still use whole mel-spectrograms.

  * •

Although diffusion loss ℒd​i​f​fsubscriptℒ𝑑𝑖𝑓𝑓\mathcal{L}_{diff} seems to converge very slowly after the beginning epochs, as shown on Figure 3, such long training is essential to get a good model because the neural network sθsubscript𝑠𝜃s_{\theta} has to learn to estimate gradients accurately for all t∈[0,1]𝑡01t\in[0,1]. Two models with almost equal diffusion losses can produce mel-spectrograms of very different quality: inaccurate predictions for a small subset S⊂[0,1]𝑆01S\subset[0,1] may have a small impact on ℒd​i​f​fsubscriptℒ𝑑𝑖𝑓𝑓\mathcal{L}_{diff} but be crucial for the output mel-spectrogram quality if ODE solver involves calculating sθsubscript𝑠𝜃s_{\theta} in at least one point belonging to S𝑆S.

Once trained, Grad-TTS enables the trade-off between quality and inference speed due to the ability to vary the number of steps N𝑁N the decoder takes to solve ODE (13) at inference. So, we evaluate four models which we denote by Grad-TTS-N where N∈[4,10,100,1000]𝑁4101001000N\in[4,10,100,1000]. We use τ=1.5𝜏1.5\tau=1.5 at synthesis for all four models. As baselines, we take an official implementation of Glow-TTS (Kim et al., 2020), the model which resembles ours to the most extent among the existing feature generators, FastSpeech (Ren et al., 2019), and state-of-the-art Tacotron2 (Shen et al., 2018). Recently proposed HiFi-GAN (Kong et al., 2020) is known to provide excellent sound quality, so we use this vocoder with all models we compare.

###  4.1 Subjective evaluation

To make subjective evaluation of TTS models, we used the crowdsourcing platform Amazon Mechanical Turk. For Mean Opinion Score (MOS) estimation we synthesized 404040 sentences from the test set with each model. The assessors were asked to estimate the quality of synthesized speech on a nine-point Likert scale, the lowest and the highest scores being 111 point (“Bad”) and 555 points (“Excellent”) with a step of 0.50.50.5 point. To ensure the reliability of the obtained results, only Master assessors were assigned to complete the listening test. Each audio was evaluated by 101010 assessors. A small subset of speech samples used in the test is available at <https://grad-tts.github.io/>.

Table 1: Ablation study of proposed generalized diffusion framework. Grad-TTS reconstructing data from 𝒩​(0,I)𝒩0𝐼\mathcal{N}(0,I) for N𝑁N reverse diffusion iterations is compared with the baseline Grad-TTS-10 – the model reconstructing data from 𝒩​(μ,I)𝒩𝜇𝐼\mathcal{N}(\mu,I) for 101010 iterations. N𝑁N | Worse, % | Identical, % | Better, %  
---|---|---|---  
101010 | 93.893.893.8 | 0.50.50.5 | 5.75.75.7  
202020 | 82.382.382.3 | 2.92.92.9 | 14.814.814.8  
505050 | 60.360.360.3 | 5.75.75.7 | 34.034.034.0  
  
MOS results with 95%percent9595\% confidence intervals are presented in Table 2. It demonstrates that although the quality of the synthesized speech gets better when we use more iterations of the reverse diffusion, the quality gain becomes marginal starting from a certain number of iterations. In particular, there is almost no difference between Grad-TTS-1000 and Grad-TTS-10 in terms of MOS, while the gap between Grad-TTS-10 and Grad-TTS-4 (444 was the smallest number of iterations leading to satisfactory quality) is much more significant. As for other feature generators, Grad-TTS-10 is competitive with all compared models, including state-of-the-art Tacotron2. Furthermore, Grad-TTS-1000 achieves almost natural synthesis with MOS being less than that for ground truth recordings by only 0.10.10.1. We would like to note that the relatively low results of FastSpeech could possibly be explained by the fact that we used its unofficial implementation <https://github.com/xcmyz/FastSpeech>.

Table 2: Model comparison. Model | Enc params111 | Dec params | RTF | Log-likelihood | MOS  
---|---|---|---|---|---  
Grad-TTS-1000 | 7.2​m7.2𝑚7.2m | 7.6​m7.6𝑚7.6m | 3.6633.6633.663 | 0.174±0.001plus-or-minus0.1740.001\mathbf{0.174\pm 0.001} | 4.44±0.05plus-or-minus4.440.05\mathbf{4.44\pm 0.05}  
Grad-TTS-100 | 0.3630.3630.363 | 4.38±0.06plus-or-minus4.380.064.38\pm 0.06  
Grad-TTS-10 | 0.0330.0330.033 | 4.38±0.06plus-or-minus4.380.064.38\pm 0.06  
Grad-TTS-4 | 0.0120.0120.012 | 3.96±0.07plus-or-minus3.960.073.96\pm 0.07  
Glow-TTS | 7.2​m7.2𝑚7.2m | 21.4​m21.4𝑚21.4m | 0.0080.0080.008 | 0.0820.0820.082 | 4.11±0.07plus-or-minus4.110.074.11\pm 0.07  
FastSpeech | 24.5​m24.5𝑚24.5m | 0.0040.004\mathbf{0.004} | −- | 3.68±0.09plus-or-minus3.680.093.68\pm 0.09  
Tacotron2 | 28.2​m28.2𝑚28.2m | 0.0750.0750.075 | −- | 4.32±0.07plus-or-minus4.320.074.32\pm 0.07  
Ground Truth | −- | −- | −- | 4.53±0.06plus-or-minus4.530.064.53\pm 0.06  
  
To verify the benefits of the proposed generalized DPM framework we trained the model with the same architecture as Grad-TTS to reconstruct mel-spectrograms from 𝒩​(0,I)𝒩0𝐼\mathcal{N}(0,I) instead of 𝒩​(μ,I)𝒩𝜇𝐼\mathcal{N}(\mu,I). The preference test provided in Table 1 shows that Grad-TTS-10 is significantly better (p<0.005𝑝0.005p<0.005 in sign test) than this model taking 101010, 202020 and even 505050 iterations of the reverse diffusion. It demonstrates that the model trained to generate from 𝒩​(0,I)𝒩0𝐼\mathcal{N}(0,I) needs more steps of ODE solver to get high-quality mel-spectrograms than Grad-TTS we propose. We believe this is because the task of reconstructing mel-spectrogram from pure noise 𝒩​(0,I)𝒩0𝐼\mathcal{N}(0,I) is more difficult than the one of reconstructing it from its noisy copy 𝒩​(μ,I)𝒩𝜇𝐼\mathcal{N}(\mu,I). One possible objection could be that the model trained with 𝒩​(0,I)𝒩0𝐼\mathcal{N}(0,I) as terminal distribution can just add μ𝜇\mu to this noise at the first step of sampling (it is possible because sθsubscript𝑠𝜃s_{\theta} has μ𝜇\mu as its input) and then repeat the same steps as our model to generate data from N​(μ,I)𝑁𝜇𝐼N(\mu,I). In this case, it would generate mel-spectrograms of the same quality as our model taking only one step more. However, this argument is wrong, since reverse diffusion removes noise not arbitrarily, but according to the reverse trajectories of the forward diffusion. Since forward diffusion adds noise gradually, reverse diffusion has to remove noise gradually as well, and the first step of the reverse diffusion cannot be adding μ𝜇\mu to Gaussian noise with zero mean because the last step of the forward diffusion is not a jump from μ𝜇\mu to zero.

Figure 4: Typical errors occurrence.

We also made an attempt to estimate what kinds of mistakes are characteristic of certain models. We compared Tacotron2, Glow-TTS, and Grad-TTS-10 as the fastest version of our model with high synthesis quality. Each record was estimated by 555 assessors. Figure 4 demonstrates the results of the multiple-choice test whose participants had to choose which kinds of errors (if any) they could hear: sonic artifacts like clicking sounds or background noise (“sonic” in the figure), mispronunciation of words/phonemes (“mispron”), unnatural pauses (“pause”), monotone speech (“monotonic”), robotic voice (“robotic”), wrong word stressing (“stress”) or others. It is clear from the figure that Glow-TTS frequently stresses words in a wrong way, and the sound it produces is perceived as “robotic” in around a quarter of cases. These are the major factors that make Glow-TTS performance inferior to that of Grad-TTS and Tacotron2, which in their turn have more or less the same drawbacks in terms of synthesis quality.

###  4.2 Objective evaluation

Although DPMs can be shown to maximize weighted variational lower bound (Ho et al., 2020) on data log-likelihood, they do not explicitly optimize exact data likelihood. In spite of this, Song et al. (2021) show that it is still possible to calculate it using the instantaneous change of variables formula (Chen et al., 2018) if we look at DPMs from the “continuous” point of view. However, it is necessary to use Hutchinson’s trace estimator to make computations feasible, so in Table 2 log-likelihood for Grad-TTS comes with a 95%percent9595\% confidence interval.

We randomly chose 505050 sentences from the test set and calculated their average log-likelihood under two probabilistic models we consider – Glow-TTS and Grad-TTS. Interestingly, Grad-TTS achieves better log-likelihood than Glow-TTS even though the latter has a decoder with 333x larger capacity and was trained to maximize exact data likelihood. Similar phenomena were observed by Song et al. (2021) in the image generation task.

###  4.3 Efficiency estimation

We assess the efficiency of the proposed model in terms of Real-Time Factor (RTF is how many seconds it takes to generate one second of audio) computed on GPU and the number of parameters. Table 2 contains efficiency information for all models under comparison. Additional information regarding absolute inference speed dependency on the input text length is given in Figure 5.

Due to its flexibility at inference, Grad-TTS is capable of real-time synthesis on GPU: if the number of decoder steps is less than 100100100, it reaches RTF <0.37absent0.37<0.37. Moreover, although it cannot compete with Glow-TTS and FastSpeech in terms of inference speed, it still can be approximately twice faster than Tacotron2 if we use 101010 decoder iterations sufficient for getting high-fidelity mel-spectrograms. Besides, Grad-TTS has around 15​m15𝑚15m parameters, thus being significantly smaller than other feature generators we compare.

Figure 5: Inference speed comparison. Text length is given in characters.

###  4.4 End-to-end TTS

The results of our preliminary experiments show that it is also possible to train an end-to-end TTS model as a DPM. In brief, we moved from U-Net to WaveGrad (Chen et al., 2021) in Grad-TTS decoder: the overall architecture resembles WaveGrad conditioned on the aligned encoder output μ𝜇\mu instead of ground truth mel-spectrograms y𝑦y as in original WaveGrad. Although synthesized speech quality is fair enough, it cannot compete with the results reported above, so we do not include our end-to-end model in the listening test but provide demo samples at <https://grad-tts.github.io/>. 11footnotetext: Encoder and duration predictor parameters are calculated together.

##  5 Future work

End-to-end speech synthesis results reported above show that it is a promising future research direction for text-to-speech applications. However, there is also much room for investigating general issues regarding DPMs.

In the analysis in Section 2, we always assume that both forward and reverse diffusion processes exist, i.e., SDEs (2) and (8) have strong solutions. It applies some Lipschitz-type constraints (Liptser & Shiryaev, 1978) on noise schedule βtsubscript𝛽𝑡\beta_{t} and, what is more important, on the neural network sθsubscript𝑠𝜃s_{\theta}. Wasserstein GANs offer an encouraging example of incorporating Lipschitz constraints into neural networks training (Gulrajani et al., 2017), suggesting that similar techniques may improve DPMs.

Little attention has been paid so far to the choice of the noise schedule βtsubscript𝛽𝑡\beta_{t} – most researchers use a simple linear schedule. Also, it is mostly unclear how to choose weights for losses (12) at time t𝑡t in the global loss function optimally. A thorough investigation of such practical questions is crucial as it can facilitate applying DPMs to new machine learning problems.

##  6 Conclusion

We have presented Grad-TTS, the first acoustic feature generator utilizing the concept of diffusion probabilistic modelling. The main generative engine of Grad-TTS is the diffusion-based decoder that transforms Gaussian noise parameterized with the encoder output into mel-spectrogram while alignment is performed with Monotonic Alignment Search. The model we propose allows to vary the number of decoder steps at inference, thus providing a tool to control the trade-off between inference speed and synthesized speech quality. Despite its iterative decoding, Grad-TTS is capable of real-time synthesis. Moreover, it can generate mel-spectrograms twice faster than Tacotron2 while keeping synthesis quality competitive with common TTS baselines.

## References

  * Anderson (1982) Anderson, B. D.  Reverse-time diffusion equation models.  _Stochastic Processes and their Applications_ , 12(3):313 – 326, 1982.  ISSN 0304-4149. 
  * Bińkowski et al. (2020) Bińkowski, M., Donahue, J., Dieleman, S., Clark, A., et al.  High Fidelity Speech Synthesis with Adversarial Networks.  In _International Conference on Learning Representations_ , 2020. 
  * Cai et al. (2020) Cai, R., Yang, G., Averbuch-Elor, H., Hao, Z., Belongie, S., Snavely, N., and Hariharan, B.  Learning Gradient Fields for Shape Generation.  In _Proceedings of the European Conference on Computer Vision (ECCV)_ , 2020. 
  * Chen et al. (2021) Chen, N., Zhang, Y., Zen, H., Weiss, R. J., Norouzi, M., and Chan, W.  WaveGrad: Estimating Gradients for Waveform Generation.  In _International Conference on Learning Representations_ , 2021. 
  * Chen et al. (2018) Chen, R. T. Q., Rubanova, Y., Bettencourt, J., and Duvenaud, D. K.  Neural Ordinary Differential Equations.  In _Advances in Neural Information Processing Systems_ , volume 31, pp. 6571–6583. Curran Associates, Inc., 2018. 
  * Donahue et al. (2021) Donahue, J., Dieleman, S., Binkowski, M., Elsen, E., and Simonyan, K.  End-to-end Adversarial Text-to-Speech.  In _International Conference on Learning Representations_ , 2021. 
  * Elias et al. (2020) Elias, I., Zen, H., Shen, J., Zhang, Y., Jia, Y., Weiss, R., and Wu, Y.  Parallel Tacotron: Non-Autoregressive and Controllable TTS, 2020. 
  * Goodfellow et al. (2014) Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y.  Generative Adversarial Nets.  In _Advances in Neural Information Processing Systems 27_ , pp. 2672–2680. Curran Associates, Inc., 2014. 
  * Gulrajani et al. (2017) Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., and Courville, A. C.  Improved Training of Wasserstein GANs.  In _Advances in Neural Information Processing Systems_ , volume 30, pp. 5767–5777. Curran Associates, Inc., 2017. 
  * Ho et al. (2020) Ho, J., Jain, A., and Abbeel, P.  Denoising Diffusion Probabilistic Models.  In _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, virtual_ , 2020. 
  * Ito (2017) Ito, K.  The LJ Speech Dataset, 2017.  URL <https://keithito.com/LJ-Speech-Dataset/>. 
  * Kim et al. (2020) Kim, J., Kim, S., Kong, J., and Yoon, S.  Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search.  In _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, virtual_ , 2020. 
  * Kingma & Dhariwal (2018) Kingma, D. P. and Dhariwal, P.  Glow: Generative flow with invertible 1x1 convolutions.  In _Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018_ , pp. 10236–10245, 2018. 
  * Kloeden & Platen (1992) Kloeden, P. E. and Platen, E.  _Numerical Solution of Stochastic Differential Equations_ , volume 23 of _Stochastic Modelling and Applied Probability_.  Springer-Verlag Berlin Heidelberg, 1992. 
  * Kong et al. (2020) Kong, J., Kim, J., and Bae, J.  HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis.  In _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, virtual_ , 2020. 
  * Kong et al. (2021) Kong, Z., Ping, W., Huang, J., Zhao, K., and Catanzaro, B.  DiffWave: A Versatile Diffusion Model for Audio Synthesis.  In _International Conference on Learning Representations_ , 2021. 
  * Kumar et al. (2019) Kumar, K., Kumar, R., de Boissiere, T., Gestin, L., et al.  MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis.  In _Advances in Neural Information Processing Systems 32_ , pp. 14910–14921. Curran Associates, Inc., 2019. 
  * Li et al. (2019) Li, N., Liu, S., Liu, Y., Zhao, S., and Liu, M.  Neural Speech Synthesis with Transformer Network.  _Proceedings of the AAAI Conference on Artificial Intelligence_ , 33:6706–6713, 07 2019. 
  * Liptser & Shiryaev (1978) Liptser, R. and Shiryaev, A.  _Statistics of Random Processes_ , volume 5 of _Stochastic Modelling and Applied Probability_.  Springer-Verlag, 1978. 
  * Luhman & Luhman (2020) Luhman, T. and Luhman, E.  Diffusion models for Handwriting Generation, 2020. 
  * Niu et al. (2020) Niu, C., Song, Y., Song, J., Zhao, S., Grover, A., and Ermon, S.  Permutation Invariant Graph Generation via Score-Based Generative Modeling.  In _AISTATS_ , 2020. 
  * Prenger et al. (2019) Prenger, R., Valle, R., and Catanzaro, B.  Waveglow: A Flow-based Generative Network for Speech Synthesis.  In _2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 3617–3621. IEEE, May 2019. 
  * Rabiner (1989) Rabiner, L.  A Tutorial on Hidden Markov Models and Selected Applications.  _Proceedings of the IEEE_ , 1989. 
  * Ren et al. (2019) Ren, Y., Ruan, Y., Tan, X., Qin, T., et al.  FastSpeech: Fast, Robust and Controllable Text to Speech.  In _Advances in Neural Information Processing Systems 32_ , pp. 3171–3180. Curran Associates, Inc., 2019. 
  * Rezende & Mohamed (2015) Rezende, D. J. and Mohamed, S.  Variational inference with normalizing flows.  In _Proceedings of the 32nd International Conference on Machine Learning, ICML 2015, Lille, France, 6-11 July 2015_ , pp. 1530–1538, 2015\. 
  * Ronneberger et al. (2015) Ronneberger, O., Fischer, P., and Brox, T.  U-Net: Convolutional Networks for Biomedical Image Segmentation.  In _Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015_ , pp. 234–241. Springer International Publishing, 2015. 
  * Shen et al. (2018) Shen, J., Pang, R., et al.  Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions.  In _2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 4779–4783, April 2018. 
  * Shen et al. (2020) Shen, J., Jia, Y., Chrzanowski, M., Zhang, Y., Elias, I., Zen, H., and Wu, Y.  Non-Attentive Tacotron: Robust and Controllable Neural TTS Synthesis Including Unsupervised Duration Modeling.  _ArXiv_ , abs/2010.04301, 2020. 
  * Sohl-Dickstein et al. (2015) Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., and Ganguli, S.  Deep Unsupervised Learning using Nonequilibrium Thermodynamics.  In _Proceedings of the 32nd International Conference on Machine Learning_ , Proceedings of Machine Learning Research, pp. 2256–2265. PMLR, 2015\. 
  * Song & Ermon (2019) Song, Y. and Ermon, S.  Generative Modeling by Estimating Gradients of the Data Distribution.  In _Advances in Neural Information Processing Systems_ , volume 32, pp. 11918–11930. Curran Associates, Inc., 2019. 
  * Song & Ermon (2020) Song, Y. and Ermon, S.  Improved Techniques for Training Score-Based Generative Models.  In _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, virtual_ , 2020. 
  * Song et al. (2021) Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B.  Score-Based Generative Modeling through Stochastic Differential Equations.  In _International Conference on Learning Representations_ , 2021. 
  * van den Oord et al. (2016) van den Oord, A., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., Kalchbrenner, N., Senior, A., and Kavukcuoglu, K.  WaveNet: A Generative Model for Raw Audio.  In _9th ISCA Speech Synthesis Workshop_ , pp. 125–125, 2016. 
  * van den Oord et al. (2018) van den Oord, A., Li, Y., et al.  Parallel WaveNet: Fast High-Fidelity Speech Synthesis.  In _Proceedings of the 35th International Conference on Machine Learning_ , volume 80 of _Proceedings of Machine Learning Research_ , pp. 3918–3926. PMLR, 10–15 Jul 2018. 
  * Yamamoto et al. (2020) Yamamoto, R., Song, E., and Kim, J.-M.  Parallel Wavegan: A Fast Waveform Generation Model Based on Generative Adversarial Networks with Multi-Resolution Spectrogram.  In _IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 6199–6203, 2020. 

## Appendix

We include an appendix with detailed derivations, proofs and additional information. Our proposed diffusion probabilistic framework employs generalized terminal distribution 𝒩​(μ,Σ)𝒩𝜇Σ\mathcal{N}(\mu,\Sigma) instead of 𝒩​(0,I)𝒩0𝐼\mathcal{N}(0,I) as proposed by Song et al. (2021). The derivation for the solution (3) of SDE (2) that transforms the original data distribution to the terminal distribution is described in Appendix A. In Appendix B we also derive the distribution which the solution (3) for the diffused data Xtsubscript𝑋𝑡X_{t} follows. Then, the goal of diffusion probabilistic modelling is to reconstruct the reverse-time trajectories of the forward diffusion process, and Song et al. (2021) showed that these dynamics can follow two different differential equations: either SDE (8) proposed by Anderson (1982) or ODE (9). So, Appendix C contains these differential equations for 𝒩​(μ,Σ)𝒩𝜇Σ\mathcal{N}(\mu,\Sigma) serving as terminal distribution. They depend on time-dependent gradient field ∇log⁡p0​t​(Xt|X0)∇subscript𝑝0𝑡conditionalsubscript𝑋𝑡subscript𝑋0\nabla\log{p_{0t}(X_{t}|X_{0})} supposed to be modelled using neural network. In order to train it, we show how to compute the gradient in Appendix D.

###  A Solving forward diffusion SDE

Forward diffusion SDE is given by

| d​Xt=12​Σ−1​(μ−Xt)​βt​d​t+βt​d​Wt,t∈[0,T],formulae-sequence𝑑subscript𝑋𝑡12superscriptΣ1𝜇subscript𝑋𝑡subscript𝛽𝑡𝑑𝑡subscript𝛽𝑡𝑑subscript𝑊𝑡𝑡0𝑇dX_{t}=\frac{1}{2}\Sigma^{-1}(\mu-X_{t})\beta_{t}dt+\sqrt{\beta_{t}}dW_{t},\ \ \ \ t\in[0,T], |  | (19)  
---|---|---|---  
  
where Xtsubscript𝑋𝑡X_{t} is n𝑛n-dimensional stochastic process, Wtsubscript𝑊𝑡W_{t} is the standard n𝑛n-dimensional Brownian motion, μ=(μ1​…​μn)𝐓𝜇superscriptsubscript𝜇1…subscript𝜇𝑛𝐓\mu=(\mu_{1}...\mu_{n})^{\mathbf{T}} is n𝑛n-dimensional vector, ΣΣ\Sigma is n×n𝑛𝑛n\times n diagonal matrix with positive diagonal elements {σi​i2}1nsuperscriptsubscriptsubscriptsuperscript𝜎2𝑖𝑖1𝑛\\{\sigma^{2}_{ii}\\}_{1}^{n} and noise schedule βtsubscript𝛽𝑡\beta_{t} is non-negative function [0,T]→ℝ+→0𝑇superscriptℝ[0,T]\rightarrow\mathbb{R}^{+}. Consider change of variables Yt=Xt−μsubscript𝑌𝑡subscript𝑋𝑡𝜇Y_{t}=X_{t}-\mu. Then we can rewrite forward diffusion SDE as

| d​Yt=−12​Σ−1​Yt​βt​d​t+βt​d​Wt.𝑑subscript𝑌𝑡12superscriptΣ1subscript𝑌𝑡subscript𝛽𝑡𝑑𝑡subscript𝛽𝑡𝑑subscript𝑊𝑡dY_{t}=-\frac{1}{2}\Sigma^{-1}Y_{t}\beta_{t}dt+\sqrt{\beta_{t}}dW_{t}. |  | (20)  
---|---|---|---  
  
For every i=1,..,ni=1,..,n we have

| d​(e12​σi​i2​∫0tβs​𝑑s​Yti)=e12​σi​i2​∫0tβs​𝑑s⋅12​σi​i2​βt​Yti​d​t+e12​σi​i2​∫0tβs​𝑑s⋅(−12​σi​i2​Yti​βt​d​t+βt​d​Wti)==e12​σi​i2​∫0tβs​𝑑s​βt​d​Wti.𝑑superscript𝑒12subscriptsuperscript𝜎2𝑖𝑖superscriptsubscript0𝑡subscript𝛽𝑠differential-d𝑠superscriptsubscript𝑌𝑡𝑖⋅superscript𝑒12subscriptsuperscript𝜎2𝑖𝑖superscriptsubscript0𝑡subscript𝛽𝑠differential-d𝑠12subscriptsuperscript𝜎2𝑖𝑖subscript𝛽𝑡superscriptsubscript𝑌𝑡𝑖𝑑𝑡⋅superscript𝑒12subscriptsuperscript𝜎2𝑖𝑖superscriptsubscript0𝑡subscript𝛽𝑠differential-d𝑠12subscriptsuperscript𝜎2𝑖𝑖superscriptsubscript𝑌𝑡𝑖subscript𝛽𝑡𝑑𝑡subscript𝛽𝑡𝑑superscriptsubscript𝑊𝑡𝑖superscript𝑒12subscriptsuperscript𝜎2𝑖𝑖superscriptsubscript0𝑡subscript𝛽𝑠differential-d𝑠subscript𝛽𝑡𝑑superscriptsubscript𝑊𝑡𝑖\begin{split}d\left(e^{\frac{1}{2\sigma^{2}_{ii}}\int_{0}^{t}{\beta_{s}ds}}Y_{t}^{i}\right)&=e^{\frac{1}{2\sigma^{2}_{ii}}\int_{0}^{t}{\beta_{s}ds}}\cdot\frac{1}{2\sigma^{2}_{ii}}\beta_{t}Y_{t}^{i}dt+e^{\frac{1}{2\sigma^{2}_{ii}}\int_{0}^{t}{\beta_{s}ds}}\cdot\left(-\frac{1}{2\sigma^{2}_{ii}}Y_{t}^{i}\beta_{t}dt+\sqrt{\beta_{t}}dW_{t}^{i}\right)=\\\ &=e^{\frac{1}{2\sigma^{2}_{ii}}\int_{0}^{t}{\beta_{s}ds}}\sqrt{\beta_{t}}dW_{t}^{i}.\end{split} |  | (21)  
---|---|---|---  
  
Exponential of a diagonal matrix is just element-wise exponential, so we can rewrite it in multidimensional form as

| d​(e12​Σ−1​∫0tβs​𝑑s​Yt)=βt​e12​Σ−1​∫0tβs​𝑑s​d​Wt⟹e12​Σ−1​∫0tβs​𝑑s​Yt−Y0=∫0tβs​e12​Σ−1​∫0sβu​𝑑u​𝑑Ws,𝑑superscript𝑒12superscriptΣ1superscriptsubscript0𝑡subscript𝛽𝑠differential-d𝑠subscript𝑌𝑡subscript𝛽𝑡superscript𝑒12superscriptΣ1superscriptsubscript0𝑡subscript𝛽𝑠differential-d𝑠𝑑subscript𝑊𝑡superscript𝑒12superscriptΣ1superscriptsubscript0𝑡subscript𝛽𝑠differential-d𝑠subscript𝑌𝑡subscript𝑌0superscriptsubscript0𝑡subscript𝛽𝑠superscript𝑒12superscriptΣ1superscriptsubscript0𝑠subscript𝛽𝑢differential-d𝑢differential-dsubscript𝑊𝑠d\left(e^{\frac{1}{2}\Sigma^{-1}\int_{0}^{t}{\beta_{s}ds}}Y_{t}\right)=\sqrt{\beta_{t}}e^{\frac{1}{2}\Sigma^{-1}\int_{0}^{t}{\beta_{s}ds}}dW_{t}\implies e^{\frac{1}{2}\Sigma^{-1}\int_{0}^{t}{\beta_{s}ds}}Y_{t}-Y_{0}=\int_{0}^{t}{\sqrt{\beta_{s}}e^{\frac{1}{2}\Sigma^{-1}\int_{0}^{s}{\beta_{u}du}}dW_{s}}, |  | (22)  
---|---|---|---  
  
or writing this down in terms of Xtsubscript𝑋𝑡X_{t}:

| Xt=e−12​Σ−1​∫0tβs​𝑑s​X0+(I−e−12​Σ−1​∫0tβs​𝑑s)​μ+∫0tβs​e−12​Σ−1​∫stβu​𝑑u​𝑑Ws,subscript𝑋𝑡superscript𝑒12superscriptΣ1superscriptsubscript0𝑡subscript𝛽𝑠differential-d𝑠subscript𝑋0𝐼superscript𝑒12superscriptΣ1superscriptsubscript0𝑡subscript𝛽𝑠differential-d𝑠𝜇superscriptsubscript0𝑡subscript𝛽𝑠superscript𝑒12superscriptΣ1superscriptsubscript𝑠𝑡subscript𝛽𝑢differential-d𝑢differential-dsubscript𝑊𝑠X_{t}=e^{-\frac{1}{2}\Sigma^{-1}\int_{0}^{t}{\beta_{s}ds}}X_{0}+\left(I-e^{-\frac{1}{2}\Sigma^{-1}\int_{0}^{t}{\beta_{s}ds}}\right)\mu+\int_{0}^{t}{\sqrt{\beta_{s}}e^{-\frac{1}{2}\Sigma^{-1}\int_{s}^{t}{\beta_{u}du}}dW_{s}}, |  | (23)  
---|---|---|---  
  
where I𝐼I is n×n𝑛𝑛n\times n identity matrix.

###  B Derivation of conditional distribution of 𝐗𝐭subscript𝐗𝐭\mathbf{X_{t}}

Let A​(s)=βs​e−12​Σ−1​∫stβu​𝑑u𝐴𝑠subscript𝛽𝑠superscript𝑒12superscriptΣ1superscriptsubscript𝑠𝑡subscript𝛽𝑢differential-d𝑢A(s)=\sqrt{\beta_{s}}e^{-\frac{1}{2}\Sigma^{-1}\int_{s}^{t}{\beta_{u}du}}. It is a diagonal matrix and its i𝑖i-th diagonal element ai​i​(s)subscript𝑎𝑖𝑖𝑠a_{ii}(s) equals βs​e−12​σi​i2​∫stβu​𝑑usubscript𝛽𝑠superscript𝑒12subscriptsuperscript𝜎2𝑖𝑖superscriptsubscript𝑠𝑡subscript𝛽𝑢differential-d𝑢\sqrt{\beta_{s}}e^{-\frac{1}{2\sigma^{2}_{ii}}\int_{s}^{t}{\beta_{u}du}}. Assume ai​i​(s)∈L2​[0,T]subscript𝑎𝑖𝑖𝑠subscript𝐿20𝑇a_{ii}(s)\in L_{2}[0,T] for each i𝑖i. Itô’s integral ∫0tai​i​(s)​𝑑Wsisuperscriptsubscript0𝑡subscript𝑎𝑖𝑖𝑠differential-dsuperscriptsubscript𝑊𝑠𝑖\int_{0}^{t}{a_{ii}(s)dW_{s}^{i}} is defined as the limit of integral sums when mesh of partition ΔΔ\Delta tends to zero:

| ∫0tai​i​(s)​𝑑Wsi=limΔ→0∑kai​i​(sk)​Δ​Wski=dlimΔ→0𝒩​(0,∑kai​i2​(sk)​Δ​sk)=d=d𝒩​(0,limΔ→0∑kai​i2​(sk)​Δ​sk)=𝒩​(0,∫0tai​i2​(s)​𝑑s),\begin{split}\int_{0}^{t}{a_{ii}(s)dW_{s}^{i}}=\lim_{\Delta\to 0}{\sum_{k}{a_{ii}(s_{k})\Delta W_{s_{k}}^{i}}}&\stackrel{{\scriptstyle d}}{{=}}\lim_{\Delta\to 0}{\mathcal{N}\left(0,\sum_{k}{a_{ii}^{2}(s_{k})\Delta s_{k}}\right)}\stackrel{{\scriptstyle d}}{{=}}\\\ &\stackrel{{\scriptstyle d}}{{=}}\mathcal{N}\left(0,\lim_{\Delta\to 0}{\sum_{k}{a_{ii}^{2}(s_{k})\Delta s_{k}}}\right)=\mathcal{N}\left(0,\int_{0}^{t}{a^{2}_{ii}(s)ds}\right),\end{split} |  | (24)  
---|---|---|---  
  
where the first equality in distribution holds due to the properties of Brownian motion and the fact that ai​i​(sk)subscript𝑎𝑖𝑖subscript𝑠𝑘a_{ii}(s_{k}) are deterministic (implying that ai​i​(sk)​Δ​Wski=ai​i​(sk)​(Wsk+1i−Wski)subscript𝑎𝑖𝑖subscript𝑠𝑘Δsuperscriptsubscript𝑊subscript𝑠𝑘𝑖subscript𝑎𝑖𝑖subscript𝑠𝑘superscriptsubscript𝑊subscript𝑠𝑘1𝑖superscriptsubscript𝑊subscript𝑠𝑘𝑖a_{ii}(s_{k})\Delta W_{s_{k}}^{i}=a_{ii}(s_{k})(W_{s_{k+1}}^{i}-W_{s_{k}}^{i}) are independent normal random variables with mean 00 and variance ai​i2​(sk)​(sk+1−sk)=ai​i2​(sk)​Δ​sksubscriptsuperscript𝑎2𝑖𝑖subscript𝑠𝑘subscript𝑠𝑘1subscript𝑠𝑘subscriptsuperscript𝑎2𝑖𝑖subscript𝑠𝑘Δsubscript𝑠𝑘a^{2}_{ii}(s_{k})(s_{k+1}-s_{k})=a^{2}_{ii}(s_{k})\Delta s_{k}) and the second equality in distribution follows from Lévy’s continuity theorem (it is easy to check that the sequence of characteristic functions of random variables on the left-hand side converges point-wise to the characteristic function of the random variable on the right-hand side). Then, simple integration gives

| ∫0tai​i2​(s)​𝑑s=∫0tβs​e−1σi​i2​∫stβu​𝑑u​𝑑s=∫0tσi​i2​d​(e−1σi​i2​∫stβu​𝑑u)=σi​i2​(1−e−1σi​i2​∫0tβs​𝑑s).superscriptsubscript0𝑡superscriptsubscript𝑎𝑖𝑖2𝑠differential-d𝑠superscriptsubscript0𝑡subscript𝛽𝑠superscript𝑒1superscriptsubscript𝜎𝑖𝑖2superscriptsubscript𝑠𝑡subscript𝛽𝑢differential-d𝑢differential-d𝑠superscriptsubscript0𝑡superscriptsubscript𝜎𝑖𝑖2𝑑superscript𝑒1superscriptsubscript𝜎𝑖𝑖2superscriptsubscript𝑠𝑡subscript𝛽𝑢differential-d𝑢superscriptsubscript𝜎𝑖𝑖21superscript𝑒1superscriptsubscript𝜎𝑖𝑖2superscriptsubscript0𝑡subscript𝛽𝑠differential-d𝑠\int_{0}^{t}a_{ii}^{2}(s)ds=\int_{0}^{t}\beta_{s}e^{-\frac{1}{\sigma_{ii}^{2}}\int_{s}^{t}\beta_{u}du}ds=\int_{0}^{t}{\sigma_{ii}^{2}d\left(e^{-\frac{1}{\sigma_{ii}^{2}}\int_{s}^{t}{\beta_{u}du}}\right)}=\sigma_{ii}^{2}\left(1-e^{-\frac{1}{\sigma_{ii}^{2}}\int_{0}^{t}{\beta_{s}ds}}\right). |  | (25)  
---|---|---|---  
  
It implies that in multidimensional case we have:

| ∫0tβs​e−12​Σ−1​∫stβu​𝑑u​𝑑Ws=∫0tA​(s)​𝑑Ws∼𝒩​(0,λ​(Σ,t)),λ​(Σ,t)=Σ​(I−e−Σ−1​∫0tβs​𝑑s),formulae-sequencesuperscriptsubscript0𝑡subscript𝛽𝑠superscript𝑒12superscriptΣ1superscriptsubscript𝑠𝑡subscript𝛽𝑢differential-d𝑢differential-dsubscript𝑊𝑠superscriptsubscript0𝑡𝐴𝑠differential-dsubscript𝑊𝑠similar-to𝒩0𝜆Σ𝑡𝜆Σ𝑡Σ𝐼superscript𝑒superscriptΣ1superscriptsubscript0𝑡subscript𝛽𝑠differential-d𝑠\int_{0}^{t}{\sqrt{\beta_{s}}e^{-\frac{1}{2}\Sigma^{-1}\int_{s}^{t}{\beta_{u}du}}dW_{s}}=\int_{0}^{t}{A(s)dW_{s}}\sim\mathcal{N}\left(0,\lambda(\Sigma,t)\right),\ \ \ \lambda(\Sigma,t)=\Sigma\left(I-e^{-\Sigma^{-1}\int_{0}^{t}{\beta_{s}ds}}\right), |  | (26)  
---|---|---|---  
  
and it follows from (23) that

| L​a​w​(Xt|X0)=𝒩​(ρ​(X0,Σ,μ,t),λ​(Σ,t)),ρ​(X0,Σ,μ,t)=e−12​Σ−1​∫0tβs​𝑑s​X0+(I−e−12​Σ−1​∫0tβs​𝑑s)​μ.formulae-sequence𝐿𝑎𝑤conditionalsubscript𝑋𝑡subscript𝑋0𝒩𝜌subscript𝑋0Σ𝜇𝑡𝜆Σ𝑡𝜌subscript𝑋0Σ𝜇𝑡superscript𝑒12superscriptΣ1superscriptsubscript0𝑡subscript𝛽𝑠differential-d𝑠subscript𝑋0𝐼superscript𝑒12superscriptΣ1superscriptsubscript0𝑡subscript𝛽𝑠differential-d𝑠𝜇Law(X_{t}|X_{0})=\mathcal{N}(\rho(X_{0},\Sigma,\mu,t),\lambda(\Sigma,t)),\ \ \ \rho(X_{0},\Sigma,\mu,t)=e^{-\frac{1}{2}\Sigma^{-1}\int_{0}^{t}{\beta_{s}ds}}X_{0}+\left(I-e^{-\frac{1}{2}\Sigma^{-1}\int_{0}^{t}{\beta_{s}ds}}\right)\mu. |  | (27)  
---|---|---|---  
  
###  C Reverse dynamics

The result by Anderson (1982) implies that if n𝑛n-dimensional process of the diffusion type Xtsubscript𝑋𝑡X_{t} satisfies

| d​Xt=f​(Xt,t)​d​t+g​(t)​d​Wt,t∈[0,T],formulae-sequence𝑑subscript𝑋𝑡𝑓subscript𝑋𝑡𝑡𝑑𝑡𝑔𝑡𝑑subscript𝑊𝑡𝑡0𝑇dX_{t}=f(X_{t},t)dt+g(t)dW_{t},\ \ \ \ \ t\in[0,T], |  | (28)  
---|---|---|---  
  
where g​(t)𝑔𝑡g(t) is a function [0,T]→ℝ→0𝑇ℝ[0,T]\rightarrow\mathbb{R}, then its reverse-time dynamics is given by

| d​Xt=(f​(Xt,t)−g2​(t)​∇log⁡pt​(Xt))​d​t+g​(t)​d​W~t,t∈[0,T],formulae-sequence𝑑subscript𝑋𝑡𝑓subscript𝑋𝑡𝑡superscript𝑔2𝑡∇subscript𝑝𝑡subscript𝑋𝑡𝑑𝑡𝑔𝑡𝑑subscript~𝑊𝑡𝑡0𝑇dX_{t}=(f(X_{t},t)-g^{2}(t)\nabla\log{p_{t}(X_{t})})dt+g(t)d\widetilde{W}_{t},\ \ \ t\in[0,T], |  | (29)  
---|---|---|---  
  
where pt​(⋅)subscript𝑝𝑡⋅p_{t}(\cdot) is the probability density function of random variable Xtsubscript𝑋𝑡X_{t} and W~tsubscript~𝑊𝑡\widetilde{W}_{t} is the reverse-time standard Brownian motion such that Xtsubscript𝑋𝑡X_{t} is independent of its past increments W~s−W~tsubscript~𝑊𝑠subscript~𝑊𝑡\widetilde{W}_{s}-\widetilde{W}_{t} for s<t𝑠𝑡s<t. Reverse-time dynamics means that all the integrals associated with reverse-time differentials have t𝑡t as their lower limit (e.g. d​Xt𝑑subscript𝑋𝑡dX_{t} relates to ∫tT𝑑Xs=XT−Xtsuperscriptsubscript𝑡𝑇differential-dsubscript𝑋𝑠subscript𝑋𝑇subscript𝑋𝑡\int_{t}^{T}{dX_{s}}=X_{T}-X_{t}). Anderson’s result is obtained under the assumption that Kolmogorov equations (for probability density functions) associated with all considered processes have unique smooth solutions. On the other hand, Song et al. (2021) argued that SDE (28) has the same forward Kolmogorov equation as the following ODE:

| d​Xt=(f​(Xt,t)−12​g2​(t)​∇log⁡pt​(Xt))​d​t,t∈[0,T],formulae-sequence𝑑subscript𝑋𝑡𝑓subscript𝑋𝑡𝑡12superscript𝑔2𝑡∇subscript𝑝𝑡subscript𝑋𝑡𝑑𝑡𝑡0𝑇dX_{t}=(f(X_{t},t)-\frac{1}{2}g^{2}(t)\nabla\log{p_{t}(X_{t})})dt,\ \ \ t\in[0,T], |  | (30)  
---|---|---|---  
  
which means that processes following (28) and (30) are equal in distribution if they start from the same initial distribution L​a​w​(X0)𝐿𝑎𝑤subscript𝑋0Law(X_{0}). In our case f​(Xt,t)=12​Σ−1​(Xt−μ)​βt𝑓subscript𝑋𝑡𝑡12superscriptΣ1subscript𝑋𝑡𝜇subscript𝛽𝑡f(X_{t},t)=\frac{1}{2}\Sigma^{-1}(X_{t}-\mu)\beta_{t} and g​(t)=βt𝑔𝑡subscript𝛽𝑡g(t)=\sqrt{\beta_{t}}, so we have two equivalent reverse diffusion dynamics:

| d​Xt=(12​Σ−1​(Xt−μ)−∇log⁡pt​(Xt))​βt​d​t+βt​d​W~t𝑑subscript𝑋𝑡12superscriptΣ1subscript𝑋𝑡𝜇∇subscript𝑝𝑡subscript𝑋𝑡subscript𝛽𝑡𝑑𝑡subscript𝛽𝑡𝑑subscript~𝑊𝑡dX_{t}=\left(\frac{1}{2}\Sigma^{-1}(X_{t}-\mu)-\nabla\log{p_{t}(X_{t})}\right)\beta_{t}dt+\sqrt{\beta_{t}}d\widetilde{W}_{t} |  | (31)  
---|---|---|---  
  
and

| d​Xt=12​(Σ−1​(Xt−μ)−∇log⁡pt​(Xt))​βt​d​t,𝑑subscript𝑋𝑡12superscriptΣ1subscript𝑋𝑡𝜇∇subscript𝑝𝑡subscript𝑋𝑡subscript𝛽𝑡𝑑𝑡dX_{t}=\frac{1}{2}\left(\Sigma^{-1}(X_{t}-\mu)-\nabla\log{p_{t}(X_{t})}\right)\beta_{t}dt, |  | (32)  
---|---|---|---  
  
where both differential equations are to be solved backwards.

###  D Score estimation

If X0subscript𝑋0X_{0} is known, then (27) implies that

| log⁡p0​t​(Xt|X0)=−n2​log⁡(2​π)−12​detλ​(Σ,t)−12​(Xt−ρ​(X0,Σ,μ,t))𝐓​λ​(Σ,t)−1​(Xt−ρ​(X0,Σ,μ,t))⟹∇log⁡p0​t​(Xt|X0)=−λ​(Σ,t)−1​(Xt−ρ​(X0,Σ,μ,t)),subscript𝑝0𝑡conditionalsubscript𝑋𝑡subscript𝑋0𝑛22𝜋12𝜆Σ𝑡12superscriptsubscript𝑋𝑡𝜌subscript𝑋0Σ𝜇𝑡𝐓𝜆superscriptΣ𝑡1subscript𝑋𝑡𝜌subscript𝑋0Σ𝜇𝑡∇subscript𝑝0𝑡conditionalsubscript𝑋𝑡subscript𝑋0𝜆superscriptΣ𝑡1subscript𝑋𝑡𝜌subscript𝑋0Σ𝜇𝑡\begin{gathered}\log{p_{0t}(X_{t}|X_{0})}=-\frac{n}{2}\log{(2\pi)-\frac{1}{2}\det{\lambda(\Sigma,t)}}-\frac{1}{2}(X_{t}-\rho(X_{0},\Sigma,\mu,t))^{\mathbf{T}}\lambda(\Sigma,t)^{-1}(X_{t}-\rho(X_{0},\Sigma,\mu,t))\implies\\\ \nabla\log{p_{0t}(X_{t}|X_{0})}=-\lambda(\Sigma,t)^{-1}(X_{t}-\rho(X_{0},\Sigma,\mu,t)),\end{gathered} |  | (33)  
---|---|---|---  
  
where p0​t(⋅|X0)p_{0t}(\cdot|X_{0}) is the probability density function of conditional distribution L​a​w​(Xt|X0)𝐿𝑎𝑤conditionalsubscript𝑋𝑡subscript𝑋0Law(X_{t}|X_{0}). So, if we sample Xtsubscript𝑋𝑡X_{t} by the formula Xt=ρ​(X0,Σ,μ,t)+ϵtsubscript𝑋𝑡𝜌subscript𝑋0Σ𝜇𝑡subscriptitalic-ϵ𝑡X_{t}=\rho(X_{0},\Sigma,\mu,t)+\epsilon_{t} where ϵt∼𝒩​(0,λ​(Σ,t))similar-tosubscriptitalic-ϵ𝑡𝒩0𝜆Σ𝑡\epsilon_{t}\sim\mathcal{N}(0,\lambda(\Sigma,t)), then ∇log⁡p0​t​(Xt|X0)=−λ​(Σ,t)−1​ϵt∇subscript𝑝0𝑡conditionalsubscript𝑋𝑡subscript𝑋0𝜆superscriptΣ𝑡1subscriptitalic-ϵ𝑡\nabla\log{p_{0t}(X_{t}|X_{0})}=-\lambda(\Sigma,t)^{-1}\epsilon_{t}. In the simplified case when Σ=IΣ𝐼\Sigma=I we have λ​(I,t)=λt​I𝜆𝐼𝑡subscript𝜆𝑡𝐼\lambda(I,t)=\lambda_{t}I where λt=1−e−∫0tβs​𝑑ssubscript𝜆𝑡1superscript𝑒superscriptsubscript0𝑡subscript𝛽𝑠differential-d𝑠\lambda_{t}=1-e^{-\int_{0}^{t}{\beta_{s}ds}}. In this case gradient of noisy data log-density reduces to ∇log⁡p0​t​(Xt|X0)=−ϵt/λt∇subscript𝑝0𝑡conditionalsubscript𝑋𝑡subscript𝑋0subscriptitalic-ϵ𝑡subscript𝜆𝑡\nabla\log{p_{0t}(X_{t}|X_{0})}=-\epsilon_{t}/\lambda_{t}. If ϵt=λt​ξtsubscriptitalic-ϵ𝑡subscript𝜆𝑡subscript𝜉𝑡\epsilon_{t}=\sqrt{\lambda_{t}}\xi_{t}, then we have

| Xt=ρ​(X0,I,μ,t)+λt​ξt,ξt∼𝒩​(0,I),∇log⁡p0​t​(Xt|X0)=−ξt/λt.formulae-sequencesubscript𝑋𝑡𝜌subscript𝑋0𝐼𝜇𝑡subscript𝜆𝑡subscript𝜉𝑡formulae-sequencesimilar-tosubscript𝜉𝑡𝒩0𝐼∇subscript𝑝0𝑡conditionalsubscript𝑋𝑡subscript𝑋0subscript𝜉𝑡subscript𝜆𝑡X_{t}=\rho(X_{0},I,\mu,t)+\sqrt{\lambda_{t}}\xi_{t},\ \ \ \xi_{t}\sim\mathcal{N}(0,I),\ \ \ \nabla\log{p_{0t}(X_{t}|X_{0})}=-\xi_{t}/\sqrt{\lambda_{t}}. |  | (34)  
---|---|---|---  
  
[◄](/html/2105.06336) [](/) [Feeling  
lucky?](/feeling_lucky) [](/land_of_honey_and_milk) [Conversion  
report](/log/2105.06337) [Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2105.06337) [View original  
on arXiv](https://arxiv.org/abs/2105.06337)[►](/html/2105.06338)

[](javascript:toggleColorScheme\(\) "Toggle ar5iv color scheme") [Copyright](https://arxiv.org/help/license) [Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Mon Mar 18 00:22:08 2024 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)
