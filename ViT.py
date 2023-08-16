import tensorflow as tf
from tensorflow.keras import layers


def preprocessing(image_size):
    return tf.keras.Sequential([
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(factor = 0.02),
        layers.RandomZoom(height_factor = 0.2, width_factor = 0.2),
    ], name = 'data_augmentation')


def CNNblock():
    return tf.keras.Sequential([
        layers.Conv2D(32, (3,3), activation = 'relu'),
        layers.Conv2D(32, (3,3), activation = 'relu')
    ])


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        
    def call(self, img):
        batch_size = tf.shape(img)[0]
        patches = tf.image.extract_patches(img, 
                                           sizes = [1, self.patch_size, self.patch_size, 1],
                                           strides = [1, self.patch_size, self.patch_size, 1],
                                           rates = [1, 1, 1, 1],
                                           padding = 'VALID')
        elements = patches.shape[-1]
        n_row, n_col = patches.shape[1], patches.shape[2]
        patches = tf.reshape(patches, [batch_size, -1, elements])
        return patches # (batch_size, total_patches, pixels_per_patch)
    

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(input_dim = num_patches, output_dim = projection_dim)

    def __call__(self, patch):
        positions = tf.range(start = 0, limit = self.num_patches, delta = 1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded # (batch_size, total_patches, embed_dims)


class TransformerEncoder(layers.Layer):
    def __init__(self, num_heads, projection_dim, epsilon = 0.001, dropout = 0.2):
        super().__init__()
        self.num_heads = num_heads
        self.projection_dim = projection_dim
        self.dropout = dropout

        self.MultiHeadAttention = layers.MultiHeadAttention(num_heads = self.num_heads, key_dim = self.projection_dim, dropout = self.dropout)
    
        self.linear_1 = layers.Dense(1024, activation = 'relu')
        self.linear_2 = layers.Dense(1024, activation = 'relu')
        self.linear_3 = layers.Dense(projection_dim, activation = 'relu')

        self.layernorm_1 = layers.LayerNormalization(epsilon = epsilon)
        self.layernorm_2 = layers.LayerNormalization(epsilon = epsilon)
  
    def feedforward(self, x):
        x = self.linear_1(x)
        x = layers.Dropout(self.dropout)(x)
        x = self.linear_2(x)
        x = layers.Dropout(self.dropout)(x)
        x = self.linear_3(x)
        return x
    
    def __call__(self, x):
        x1 = self.layernorm_1(x)
        attn_output = self.MultiHeadAttention(x1, x1)
        x2 = x + attn_output
        x3 = self.layernorm_2(x2)
        x3 = self.feedforward(x3)
        output = x2 + x3
        return output

class ViT_Transformer(tf.keras.Model):
    def __init__(self, num_heads, 
                 num_layers, 
                 patch_size, 
                 num_patches, 
                 num_class, 
                 image_size, 
                 projection_dim, 
                 epsilon = 0.001, 
                 dropout = 0.2): 
        
        super().__init__()
        self.num_heads = num_heads
        self.num_class = num_class
        self.num_layers = num_layers
        self.projection_dim = projection_dim
        self.dropout = dropout
        self.epsilon = epsilon

        self.preprocessing = preprocessing(image_size)
        self.CNNblock = CNNblock()
        self.Patches = Patches(patch_size)
        self.PatchEncoder = PatchEncoder(num_patches, projection_dim)
        self.TransformerEncoderLayer = [TransformerEncoder(num_heads, projection_dim) for _ in range(self.num_layers)]

        self.layernorm = layers.LayerNormalization(epsilon = epsilon)
        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(dropout)
        self.linear = layers.Dense(512, activation = 'relu')
        self.ffn = layers.Dense(num_class, activation = 'softmax')
        
    
    def call(self, inputs):
        inputs = self.preprocessing(inputs)
        inputs = self.CNNblock(inputs)
        patches = self.Patches(inputs)
        encoded_patches = self.PatchEncoder(patches)
        x = encoded_patches
        
        for i in range(self.num_layers):
            x = self.TransformerEncoderLayer[i](x)

        representation = self.layernorm(x)
        representation = self.flatten(representation)
        representation = self.dropout(representation)
        features = self.linear(representation)
        output = self.ffn(features)
        return output
