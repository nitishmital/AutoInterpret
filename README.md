# Coin
Chain of Interpretability

First we adapt the original MAIA implementation from the paper - https://arxiv.org/abs/2404.14394, to operate it with Google Gemini.
Second we notice that some neurons can be polysemantic. Therefore, we train sparse autoencoders (SAE) to resolve the polysemanticity into monosemantic neurons. 

More information on the implementation can be found inside the maia_gemini folder, copied over from the original codebase - https://github.com/multimodal-interpretability/maia. 
