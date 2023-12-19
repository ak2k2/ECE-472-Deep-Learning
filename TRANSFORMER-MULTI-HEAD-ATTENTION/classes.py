import numpy as np
import tensorflow as tf


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # matmul_qk
    depth = tf.cast(tf.shape(k)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # Add the mask to the scaled tensor.
    if mask is not None:
        logits += mask * -1e9

    # Softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(logits, axis=-1)

    output = tf.matmul(attention_weights, v)
    return output


def positional_encoding(position, d_model):
    angle_rates = 1 / np.power(
        10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model)
    )
    angle_rads = np.arange(position)[:, np.newaxis] * angle_rates

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class MultiHeadAttention(tf.Module):
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.Variable(tf.random.normal([d_model, d_model]))
        self.wk = tf.Variable(tf.random.normal([d_model, d_model]))
        self.wv = tf.Variable(tf.random.normal([d_model, d_model]))
        self.dense = tf.Variable(tf.random.normal([d_model, d_model]))

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def __call__(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        print("After splitting heads:", q.shape, k.shape, v.shape)

        scaled_attention = scaled_dot_product_attention(q, k, v, mask)
        print("After scaled dot product attention:", scaled_attention.shape)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        print("After concatenating heads:", concat_attention.shape)

        output = tf.matmul(concat_attention, self.dense)
        return output


def pointwise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(dff, activation="relu"),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model),  # (batch_size, seq_len, d_model)
        ]
    )


class TransformerBlock(tf.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = pointwise_feed_forward_network(d_model, dff)

        self.layernorm1_gamma = tf.Variable(tf.ones([d_model]))
        self.layernorm1_beta = tf.Variable(tf.zeros([d_model]))
        self.layernorm2_gamma = tf.Variable(tf.ones([d_model]))
        self.layernorm2_beta = tf.Variable(tf.zeros([d_model]))

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def layer_norm(self, x, gamma, beta, epsilon=1e-6):
        mean, variance = tf.nn.moments(x, axes=[-1], keepdims=True)
        normalized = (x - mean) / tf.sqrt(variance + epsilon)
        result = gamma * normalized + beta
        print("After layer normalization:", result.shape)
        return result

    def __call__(self, x, training, mask):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm(
            x + attn_output, self.layernorm1_gamma, self.layernorm1_beta
        )

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layer_norm(
            out1 + ffn_output, self.layernorm2_gamma, self.layernorm2_beta
        )

        return out2


class EncoderLayer(tf.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = pointwise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def __call__(self, x, training, mask):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        out2 = self.layernorm2(out1 + ffn_output)
        print("EncoderLayer output:", out2.shape)
        return out2


class DecoderLayer(tf.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = pointwise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def __call__(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1 = self.mha1(x, x, x, look_ahead_mask)  # Only one value is unpacked
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        print("DecoderLayer output:", out3.shape)
        return out3


class Encoder(tf.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        maximum_position_encoding,
        rate=0.1,
    ):
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(rate)

    def __call__(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # Adding embedding
        x = self.embedding(x)  # Ensure this is [batch_size, seq_len, d_model]

        # Debugging: Print shapes after embedding
        print("Shape of x after embedding:", x.shape)

        # Scaling
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # Debugging: Print shapes after scaling
        print("Shape of x after scaling:", x.shape)

        x += self.pos_encoding[:, :seq_len, :]
        print("Shape of x after adding positional encoding:", x.shape)

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x


class Decoder(tf.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        target_vocab_size,
        maximum_position_encoding,
        rate=0.1,
    ):
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)

    def __call__(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        print("Decoder After adding positional encoding:", x.shape)

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.dec_layers[i](
                x, enc_output, training, look_ahead_mask, padding_mask
            )

        return x


class Transformer(tf.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        target_vocab_size,
        pe_input,
        pe_target,
        rate=0.1,
    ):
        self.encoder = Encoder(
            num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate
        )

        self.decoder = Decoder(
            num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate
        )

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def __call__(
        self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask
    ):
        enc_output = self.encoder(inp, training, enc_padding_mask)

        # Adjusted to receive only the decoder output
        dec_output = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask
        )

        final_output = self.final_layer(dec_output)
        return final_output
