from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, Activation, concatenate,Dropout
from tensorflow.keras.models import Model

class UNet(Model):
    def __init__(self, image_height, image_width):
        super(UNet, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        
    def conv_block(self, n, input_tensor):
        x = Conv2D(n, 3, padding='same', kernel_initializer='he_normal')(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        x = Conv2D(n, 3, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        print("conv_block output shape:", x.shape)  # Debug print
        return x

    def deconv_block(self, n, input_tensor):
        x = Conv2DTranspose(n, (2, 2), strides=2, padding="same")(input_tensor)
        print("deconv_block output shape:", x.shape)  # Debug print
        return x

    def encoder(self, n, input_tensor):
        x = self.conv_block(n, input_tensor)
        y = MaxPooling2D(pool_size=(2, 2))(x)
        print("encoder output shape after pooling:", y.shape)  # Debug print
        return x, y

    def decoder(self, n, input_tensor, concat_tensor):
        y = self.deconv_block(n, input_tensor)
        print("decoder output shape after deconv_block:", y.shape)  # Debug print
        print("concat_tensor shape for concatenation:", concat_tensor.shape)  # Debug print
        y = concatenate([y, concat_tensor], axis=3)
        y = self.conv_block(n, y)
        y = Dropout(0.2)(y)

        return y

    def model(self, output_classes, filters=64):
        # Encoder
        input_layer = Input(shape=(self.image_height, self.image_width, 3), name='image_input')
        l1, l2_input = self.encoder(filters, input_layer)
        l2, l3_input = self.encoder(filters * 2, l2_input)
        l3, l4_input = self.encoder(filters * 4, l3_input)
        l4, l5_input = self.encoder(filters * 8, l4_input)

        # Bridge
        bridge = self.conv_block(filters * 16, l5_input)
        bridge = Dropout(0.2)(bridge)


        # Decoder
        d4 = self.decoder(filters * 8, bridge, l4)
        d3 = self.decoder(filters * 4, d4, l3)
        d2 = self.decoder(filters * 2, d3, l2)
        d1 = self.decoder(filters, d2, l1)

        # Output Layer
        output_layer = Conv2D(output_classes, (1, 1))(d1)
        output_layer = BatchNormalization()(output_layer)
        output_layer = Activation('softmax')(output_layer)

        model = Model(inputs=input_layer, outputs=output_layer, name='Unet')
        return model
