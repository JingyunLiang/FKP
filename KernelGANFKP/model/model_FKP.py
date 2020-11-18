import torch
from .networks import Generator_KP, Discriminator_KP, weights_init_G_KP, weights_init_D_KP
from .loss import calc_gradient_penalty, SphericalOptimizer
import sys
sys.path.append('../')
from util import save_final_kernel_png, move2cpu, analytic_kernel, kernel_shift

'''
# ------------------------------------------
# model of KernelGAN-FKP
# ------------------------------------------
'''


class KernelGAN_FKP:
    '''
    # ------------------------------------------
    # (1) create model, loss and optimizer
    # ------------------------------------------
    '''
    def __init__(self, conf):
        # Acquire configuration
        self.conf = conf

        # Define the GAN
        self.G = Generator_KP(conf).cuda()
        self.d_input_shape = self.G.output_size
        self.D = Discriminator_KP(conf, self.d_input_shape).cuda()
        self.d_output_shape = self.D.output_size

        # Initialize networks weights, for G, do not initialize NF
        self.G.apply(weights_init_G_KP)
        self.D.apply(weights_init_D_KP)

        # Optimizers
        self.optimizer_G = SphericalOptimizer(conf.G_kernel_size, torch.optim.Adam, [self.G.kernel_code], lr=conf.g_lr,
                                              betas=(conf.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=conf.d_lr, betas=(conf.beta1, 0.999))

        print('*' * 60 + '\nSTARTED KernelGAN-FKP on: \"%s\"...' % conf.input_image_path)

    '''
    # ---------------------
    # (2) training
    # ---------------------
    '''

    def train(self, g_input, d_input, iteration):
        self.g_input = g_input.contiguous().cuda()
        self.d_input = d_input.contiguous().cuda()

        for _ in range(3):
            loss_d_fake, loss_d_real = self.train_d()
        for _ in range(3):
            loss_g = self.train_g()

        if (iteration % 10 == 0 or iteration == 1) and self.conf.verbose:
            save_final_kernel_png(move2cpu(self.curr_k), self.conf, self.conf.kernel_gt, iteration)
            print('\n Iter {}, D_loss_fake: {}, D_loss_real: {}, G_loss: {}'.format(iteration, loss_d_fake.data,
                                                                                    loss_d_real.data, loss_g.data))

    '''
    # ---------------------
    # (2.1) training of D
    # ---------------------
    '''

    def train_d(self):
        self.optimizer_D.zero_grad()

        # real
        d_pred_real = self.D.forward(self.d_input)
        loss_d_real = -d_pred_real.mean()
        loss_d_real.backward(retain_graph=True)

        # fake
        self.g_output, self.curr_k = self.G.forward(self.g_input)
        # self.g_output += torch.randn_like(self.g_output) / 255.
        d_pred_fake = self.D.forward(self.g_output.detach())
        loss_d_fake = d_pred_fake.mean()
        loss_d_fake.backward(retain_graph=True)

        # GP
        gradient_penalty = calc_gradient_penalty(self.D, self.d_input, self.g_output, 0.1, self.d_input.device)  # 0.1
        gradient_penalty.backward()

        self.optimizer_D.step()
        return loss_d_fake, loss_d_real

    '''
    # ---------------------
    # (2.2) training of G
    # ---------------------
    '''

    def train_g(self):
        self.optimizer_G.opt.zero_grad()

        d_pred_fake = self.D.forward(self.g_output)
        loss_g = -d_pred_fake.mean()
        loss_g.backward(retain_graph=True)

        self.optimizer_G.step()
        return loss_g

    '''
    # ---------------------
    # (3) finish
    # ---------------------
    '''

    def finish(self):
        save_final_kernel_png(move2cpu(self.curr_k), self.conf, self.conf.kernel_gt)
        if self.conf.verbose:
            print(
                'KernelGAN-FKP estimation complete! (see --%s-- folder)\n' % self.conf.output_dir_path + '*' * 60 + '\n\n')

        k_2 = move2cpu(self.curr_k)
        if self.conf.X4:
            k_4 = analytic_kernel(k_2)
            k_4 = kernel_shift(k_4, 4)
            return k_4
        else:
            return k_2



