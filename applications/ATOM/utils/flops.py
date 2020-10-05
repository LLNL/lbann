def encoder_fp_flops(seq_len, embed_dim, hidden_dim, latent_dim):
    """Estimate FLOPs/sample in ATOM VAE encoder forward prop"""
    flops = 0

    # RNN
    flops += 2 * (3*hidden_dim*embed_dim + 3*hidden_dim**2) * seq_len
    flops += 2 * (3*hidden_dim) * seq_len

    # FC
    flops += 2 * (latent_dim*hidden_dim) + latent_dim

    return flops

def decoder_fp_flops(seq_len, vocab_size, embed_dim, latent_dim, hidden_dim):
    """Estimate FLOPs/sample in ATOM VAE decoder forward prop"""
    flops = 0

    # FC (init hidden)
    flops += 2 * (hidden_dim*latent_dim) + hidden_dim

    # RNN 1
    flops += 2 * (3*hidden_dim*(embed_dim+latent_dim) + 3*hidden_dim**2) * seq_len
    flops += 2 * (3*hidden_dim) * seq_len

    # RNN 2
    flops += 2 * (3*hidden_dim**2 + 3*hidden_dim**2) * seq_len
    flops += 2 * (3*hidden_dim) * seq_len

    # RNN 3
    flops += 2 * (3*hidden_dim**2 + 3*hidden_dim**2) * seq_len
    flops += 2 * (3*hidden_dim) * seq_len

    # FC (preds)
    flops += 2 * (vocab_size*hidden_dim) * seq_len
    flops += vocab_size * seq_len

    return flops

def discriminator_fp_flops(discriminator_dims, seq_len, embed_dim, latent_dim):
    flops = 0

    # FC 1
    flops += 2 * discriminator_dims[0] * ((embed_dim+latent_dim)*seq_len)
    flops += discriminator_dims[0]

    # Remaining FCs
    for i in range(1, len(discriminator_dims)):
        flops += 2 * discriminator_dims[i] * discriminator_dims[i-1]
        flops += discriminator_dims[i]

    return flops

if __name__ == "__main__":

    # Encoder
    flops = 3 * encoder_fp_flops(
        seq_len=100,
        embed_dim=40,
        hidden_dim=256,
        latent_dim=128,
    )

    # Decoder
    flops += 3 * decoder_fp_flops(
        seq_len=100,
        vocab_size=40,
        embed_dim=40,
        latent_dim=128,
        hidden_dim=512,
    )

    # Discriminator
    # Note: Called 3 times, one time without weight gradients
    flops += 8 * discriminator_fp_flops(
        discriminator_dims=[128, 64, 1],
        seq_len=100,
        embed_dim=40,
        latent_dim=128,
    )

    # Print PFLOPs
    print("PFLOPs/sample = ",flops/1e15)
