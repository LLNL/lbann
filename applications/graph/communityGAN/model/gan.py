import lbann

import model.generator
import model.discriminator
from util import str_list

class CommunityGAN(lbann.modules.Module):

    def __init__(
            self,
            num_vertices,
            motif_size,
            embed_dim,
            discriminator_learn_rate,
            generator_learn_rate,
            generator_type='greedy',
            embeddings_device='CPU',
            initial_embeddings=None,
    ):
        super().__init__()
        self.num_vertices = num_vertices
        self.embed_dim = embed_dim
        self.discriminator_learn_rate = discriminator_learn_rate
        self.generator_learn_rate = generator_learn_rate

        # Construct generator
        if generator_type == 'greedy':
            self.generator = model.generator.GreedyGenerator(
                num_vertices,
                embed_dim,
                generator_learn_rate,
                embeddings_device=embeddings_device,
                initial_embeddings=initial_embeddings,
            )
        elif generator_type == 'trivial':
            self.generator = model.generator.TrivialGenerator(
                num_vertices,
                motif_size,
                embed_dim,
                generator_learn_rate,
                embeddings_device=embeddings_device,
                initial_embeddings=initial_embeddings,
            )
        else:
            raise ValueError(f'Unrecognized generator type ({generator_type})')

        # Construct discriminator
        self.discriminator = model.discriminator.Discriminator(
            num_vertices,
            motif_size,
            embed_dim,
            discriminator_learn_rate,
            embeddings_device=embeddings_device,
            initial_embeddings=initial_embeddings,
        )

    def forward(
            self,
            motif_indices,
            motif_size,
            walk_indices,
            walk_length,
    ):

        # Apply generator
        fake_motif_indices, gen_prob, gen_log_prob = self.generator(
            walk_length,
            walk_indices,
            motif_size,
        )

        # Get discriminator embeddings in log-space
        all_motif_indices = lbann.Concatenation(
            motif_indices,
            fake_motif_indices,
            device='CPU',
        )
        all_motif_embeddings = self.discriminator.get_embeddings(all_motif_indices)
        all_motif_log_embeddings = lbann.Log(all_motif_embeddings)
        all_motif_log_embeddings = lbann.Slice(
            all_motif_log_embeddings,
            slice_points=str_list([0, motif_size, 2*motif_size]),
        )
        real_motif_log_embeddings = lbann.Identity(all_motif_log_embeddings)
        fake_motif_log_embeddings = lbann.Identity(all_motif_log_embeddings)

        # Apply discriminator
        real_disc_prob, real_disc_log_not_prob \
            = self.discriminator(motif_size, real_motif_log_embeddings)
        fake_disc_prob, fake_disc_log_not_prob \
            = self.discriminator(motif_size, fake_motif_log_embeddings)

        # Loss function
        # L_disc = - log(D(real)) - log(1-D(fake))
        # L_gen = log(G) * stop_gradient(log(1-D(fake)))
        real_disc_log_prob \
            = lbann.Log(lbann.Clamp(real_disc_prob, min=1e-37, max=1))
        disc_loss = lbann.WeightedSum(
            real_disc_log_prob,
            fake_disc_log_not_prob,
            scaling_factors=str_list([-1,-1]),
        )
        gen_loss = lbann.Multiply(
            gen_log_prob,
            lbann.StopGradient(fake_disc_log_not_prob),
        )
        loss = lbann.Add(disc_loss, gen_loss)

        return loss, real_disc_prob, fake_disc_prob, gen_prob
