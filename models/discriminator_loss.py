import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd import grad
from src import utils



class DiscriminatorLoss(nn.Module):

    def __init__(self, opt, dis):
        super(DiscriminatorLoss, self).__init__()

        self.dis = dis
        self.gpu_id = opt.gpu_ids[0]

        # Adversarial criteria for the predictions
        self.crits = []

        # Targets for criteria
        self.labels_real = []
        self.labels_fake = []

        # Iterate over discriminators to inialize criteria
        for loss_type, output_shapes in zip(dis.adv_loss_types, dis.shapes):

            if loss_type == 'gan' or loss_type == 'lsgan':

                # Set criteria
                if loss_type == 'gan':
                    self.crits += [nn.BCEWithLogitsLoss()]
                elif loss_type == 'lsgan':
                    self.crits += [nn.MSELoss()]
                
                # Set labels
                labels_real = []
                labels_fake = []

                for shape in output_shapes:

                    labels_real += [Variable(torch.ones(shape).cuda(self.gpu_id))]
                    labels_fake += [Variable(torch.zeros(shape).cuda(self.gpu_id))]

                self.labels_real += [labels_real]
                self.labels_fake += [labels_fake]

            elif loss_type == 'wgan':

                self.crits += [None]

                self.labels_real += [None]
                self.labels_fake += [None]

        # Initialize criterion for aux loss
        self.crit_aux = utils.get_criterion(opt.aux_loss_type)
        #self.gen_aux_loss_weight = opt.gen_aux_loss_weight
        #self.dis_aux_loss_weight = opt.dis_aux_loss_weight

    def calc_gradient_penalty(
        self,  
        img_real_dst, 
        img_fake_dst, 
        img_real_src,
        enc,
        dis_idx):

        # Calculate weights for interpolation
        alpha = torch.rand(img_real_dst.data.size(0), 1, 1, 1)
        alpha = alpha.expand(img_real_dst.data.size())
        alpha = alpha.cuda(self.gpu_id)

        # Interpolate image from domain B
        image_dst = alpha * img_real_dst.data + (1-alpha) * img_fake_dst.data
        image_dst = Variable(image_dst, requires_grad=True)

        # Store for grad
        input = image_dst

        # Pass pair from domain A in case of aligned problem
        if img_real_src is not None:
            image_src = Variable(img_real_src, requires_grad=True)
            input = torch.cat([input, image_src], 0)
        else:
            image_src = None

        # Calculate predictions
        outputs, _ = self.dis(image_dst, image_src, enc, dis_idx)

        outputs = [o for dis_outputs in outputs for o in dis_outputs]
        shapes = [o.shape for o in outputs]
        weights = [w for i in dis_idx for w in self.dis.weights[i]]

        grad_outputs = [torch.ones(s).cuda(self.gpu_id) / s[1] * w for s, w in zip(shapes, weights)]

        # Calculate gradients with respect to input            
        gradients = grad(
            outputs=outputs, inputs=input,
            grad_outputs=grad_outputs,
            create_graph=True, retain_graph=True, only_inputs=True)[0]

        # Reshape gradients for image_src to channels (if present)
        gradients = gradients.view(input.size(0), -1)

        # Calculate gradient penalty
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10

        return gradient_penalty

    def __call__(
        self, 
        img_real_dst, 
        img_fake_dst=None, 
        aux_real_dst=None, 
        img_real_src=None,
        enc=None):

        # Preds for real (during dis backprop) or fake (during gen backprop)
        outputs_real, outputs_aux_real = self.dis(
            img_real_dst, 
            img_real_src, 
            enc)

        # Preds for fake during dis backprop
        if img_fake_dst is not None:
            outputs_fake, _ = self.dis(
                img_fake_dst, 
                img_real_src, 
                enc)

        # Losses
        loss = 0
        losses_adv = []
        losses_aux = []

        dis_idx = []

        # Iterate over outputs of each discriminator
        for i, crit in enumerate(self.crits):
        
            # Losses for each discriminator output
            losses_adv_i = []

            for j in range(len(outputs_real[i])):
                
                # GAN, LSGAN or WGAN loss
                if crit is not None:

                    losses_adv_i += [crit(
                        outputs_real[i][j], 
                        self.labels_real[i][j])]

                    if img_fake_dst is not None:

                        losses_adv_i[-1] += crit(
                            outputs_fake[i][j], 
                            self.labels_fake[i][j])

                        losses_adv_i[-1] *= 0.5

                    losses_adv_i[-1] *= self.dis.weights[i][j]

                    loss += losses_adv_i[-1]

                else:

                    losses_adv_i += [outputs_real[i][j].mean()]

                    if img_fake_dst is not None:

                        losses_adv_i[-1] -= outputs_fake[i][j].mean()

                        dis_idx += [i]

                    losses_adv_i[-1] *= self.dis.weights[i][j]

                    loss -= losses_adv_i[-1]

            # Get loss values
            losses_adv_i = [loss_adv.data.item() for loss_adv in losses_adv_i]

            # TODO: make plots for each output separately
            losses_adv += [sum(losses_adv_i)]

            # Auxiliary loss
            #if (self.dis_aux_loss_weight and
            #   (img_fake_dst is not None or
            #    img_fake_dst is None and not self.gen_aux_loss_weight)):

            #    loss_aux = self.crit_aux(outputs_aux_real[i], aux_real_dst)
            #    loss_aux *= self.dis_aux_loss_weight

            #    losses_aux += [loss_aux.data.item()]

            #    loss += loss_aux

        if dis_idx and self.training:

            loss += self.calc_gradient_penalty(
                img_real_dst=img_real_dst, 
                img_fake_dst=img_fake_dst,
                img_real_src=img_real_src,
                enc=enc,
                dis_idx=dis_idx)

        return loss, losses_adv, losses_aux
