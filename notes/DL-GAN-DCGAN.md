# DL-GAN-DCGAN

## GAN

Random(uniform noise -1, 1) -> Conv (leaky relu) -> tanh (+mnist input) -> conv (leaky) -> sigmoid

there are three models: g_model, d_model_real(mnist), d_model_fake(g_model output)

g_loss = g_model loss

d_loss = d_model_real_loss + d_model_fake_loss

#### DCGAN

d: conv(LeNeT)

g: de_conv + bn

de_conv 使用upsamling, nn插值

g_loss/g_loss与GAN相同

