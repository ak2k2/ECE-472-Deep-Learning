import tensorflow as tf

from classes import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    MultiHeadAttention,
    Transformer,
    TransformerBlock,
    positional_encoding,
    scaled_dot_product_attention,
)


def test_scaled_dot_product_attention():
    # Create a mock query, key, value tensors
    temp_k = tf.constant(
        [[10, 0, 0], [0, 10, 0], [0, 0, 10], [0, 0, 10]], dtype=tf.float32
    )  # (4, 3)
    temp_v = tf.constant(
        [[1, 0], [10, 0], [100, 5], [1000, 6]], dtype=tf.float32
    )  # (4, 2)
    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)

    attention_output = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)

    assert attention_output.shape == (1, 2)


def test_multihead_attention_split_heads():
    mha = MultiHeadAttention(d_model=512, num_heads=8)
    sample_input = tf.random.uniform((1, 60, 512))  # (batch_size, seq_len, d_model)

    # Split heads
    split_heads_output = mha.split_heads(sample_input, batch_size=1)
    assert split_heads_output.shape == (
        1,
        8,
        60,
        64,
    )  # (batch_size, num_heads, seq_len, depth)


def test_multihead_attention():
    mha = MultiHeadAttention(d_model=512, num_heads=8)
    v = k = q = tf.random.uniform((1, 60, 512))  # (batch_size, seq_len, d_model)

    # Apply multi-head attention
    output = mha(v, k, q, mask=None)
    assert output.shape == (1, 60, 512)  # (batch_size, seq_len, d_model)


def test_transformer_block_output_shape():
    sample_size = 64
    seq_length = 10
    feature_size = 512

    transformer_block = TransformerBlock(feature_size, num_heads=8, dff=2048)
    input_sequence = tf.random.uniform((sample_size, seq_length, feature_size))

    output = transformer_block(input_sequence, training=False, mask=None)
    assert output.shape == (sample_size, seq_length, feature_size)


def test_transformer_block_internal_components():
    transformer_block = TransformerBlock(d_model=512, num_heads=8, dff=2048)
    sample_input = tf.random.uniform((64, 43, 512))  # (batch_size, seq_len, d_model)

    assert isinstance(transformer_block.mha, MultiHeadAttention)
    assert (
        len(transformer_block.ffn.layers) > 1
    )  # Should have multiple layers in the FFN


def test_positional_encoding_shape():
    seq_length = 100
    d_model = 512
    pos_encoding = positional_encoding(seq_length, d_model)
    assert pos_encoding.shape == (1, seq_length, d_model)


def test_encoder_layer_output_shape():
    sample_input = tf.random.uniform((64, 43, 512))
    encoder_layer = EncoderLayer(d_model=512, num_heads=8, dff=2048)
    output = encoder_layer(sample_input, training=False, mask=None)
    assert output.shape == sample_input.shape


def test_decoder_layer_output_shape():
    sample_input = tf.random.uniform((64, 50, 512))
    enc_output = tf.random.uniform((64, 43, 512))
    decoder_layer = DecoderLayer(d_model=512, num_heads=8, dff=2048)
    output = decoder_layer(
        sample_input,
        enc_output,
        training=False,
        look_ahead_mask=None,
        padding_mask=None,
    )
    assert output.shape == sample_input.shape


def test_encoder_output_shape():
    sample_input = tf.random.uniform((64, 512))
    encoder = Encoder(
        num_layers=2,
        d_model=512,
        num_heads=8,
        dff=2048,
        input_vocab_size=8500,
        maximum_position_encoding=1000,
    )
    output = encoder(sample_input, training=False, mask=None)
    print("-" * 50)
    print(output.shape)
    print("-" * 50)
    assert output.shape == (64, 512, 512)


def test_decoder_output_shape():
    batch_size = 64
    target_seq_len = 50  # Adjust as needed
    d_model = 512

    # Prepare sample input for the decoder
    sample_input = tf.random.uniform((batch_size, target_seq_len), dtype=tf.float32)

    # Mock encoder output
    enc_output = tf.random.uniform((batch_size, target_seq_len, d_model))

    # Look-ahead mask and padding mask (you can generate or provide dummy values)
    look_ahead_mask = None
    padding_mask = None

    # Instantiate the decoder
    decoder = Decoder(
        num_layers=2,
        d_model=d_model,
        num_heads=8,
        dff=2048,
        target_vocab_size=8000,
        maximum_position_encoding=5000,
    )

    # Get the output from the decoder
    output = decoder(
        sample_input,
        enc_output,
        training=False,
        look_ahead_mask=look_ahead_mask,
        padding_mask=padding_mask,
    )

    print("-" * 50)
    print("Decoder output shape:", output.shape)
    print("-" * 50)
    assert output.shape == (batch_size, target_seq_len, d_model)


def test_transformer_output_shape():
    batch_size = 64
    input_seq_len = 43  # Length of the input sequence
    target_seq_len = 50  # Length of the target sequence

    # Input and target sequences should be token indices
    sample_input = tf.random.uniform(
        (batch_size, input_seq_len), maxval=8500, dtype=tf.int32
    )
    target = tf.random.uniform(
        (batch_size, target_seq_len), maxval=8000, dtype=tf.int32
    )

    transformer = Transformer(
        num_layers=2,
        d_model=512,
        num_heads=8,
        dff=2048,
        input_vocab_size=8500,
        target_vocab_size=8000,
        pe_input=1000,
        pe_target=5000,
    )

    output = transformer(
        sample_input,
        target,
        training=False,
        enc_padding_mask=None,
        look_ahead_mask=None,
        dec_padding_mask=None,
    )

    assert output.shape == (batch_size, target_seq_len, 8000)


if __name__ == "__main__":
    test_scaled_dot_product_attention()
    test_multihead_attention_split_heads()
    test_multihead_attention()
    test_transformer_block_output_shape()
    test_transformer_block_internal_components()
    test_positional_encoding_shape()
    test_encoder_layer_output_shape()
    test_decoder_layer_output_shape()
    test_encoder_output_shape()
    test_decoder_output_shape()
    test_transformer_output_shape()
    print("All tests passed")
