def encoder_fp_flops(seq_len, embed_dim, hidden_dim, latent_dim):
    """Estimate FLOPs/sample in ATOM VAE encoder forward prop"""
    flops = 0

    # RNN
    flops += 2 * (3*hidden_dim*embed_dim + 3*hidden_dim**2) * seq_len

    # 2 FCs (mu, logvar)
    flops += 2 * 2 * (latent_dim*hidden_dim)

    return flops

def decoder_fp_flops(seq_len, vocab_size, embed_dim, latent_dim, hidden_dim):
    """Estimate FLOPs/sample in ATOM VAE decoder forward prop"""
    flops = 0

    # FC (init hidden)
    flops += 2 * (hidden_dim*latent_dim)

    # RNN 1
    flops += 2 * (3*hidden_dim*(embed_dim+latent_dim) + 3*hidden_dim**2) * seq_len

    # RNN 2-3
    flops += 2 * 2 * (3*hidden_dim**2 + 3*hidden_dim**2) * seq_len

    # FC (preds)
    flops += 2 * (vocab_size*hidden_dim) * seq_len

    return flops

if __name__ == "__main__":

    # Forward prop
    flops = encoder_fp_flops(seq_len=100, embed_dim=40, hidden_dim=256, latent_dim=128)
    flops += decoder_fp_flops(seq_len=100, vocab_size=40, embed_dim=40, latent_dim=128, hidden_dim=512)

    # Backprop
    flops *= 3

    # Print PFLOPs
    print("PFLOPs/sample = ",flops/1e15)
