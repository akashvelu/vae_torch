import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

def calculate_padding(img_dim, target_dim, filter_size, stride):
    """
    Calculate the amount of padding neeeded in conv layers to transform image of dimension img_dim x img_dim into image
    of dimension target_dim x target_dim, when using a kernel of size filter_size and a stride specified by stride

    :param img_dim (int): source image dimension (image is square, of spacial dimension img_dim x img_dim)
    :param target_dim (int): output target dimension (target_dim x target_dim)
    :param filter_size (int): kernel dimension used in the convolutional layer
    :param stride (int): stride used in the convolutional layer
    :return padding (int): padding to be used in the convolutional layer
    """

    padding = ((target_dim - 1) * stride - img_dim + filter_size) / 2
    return int(padding)

class VAE(nn.Module):
    """
    (Convolutional) Variational Auto-Encoder trained on the MNIST dataset
    """
    def __init__(self, img_dim, img_channels, latent_dim, enc_conv_neurons=None, dec_conv_neurons=None):
        """
        :param img_dim (int): spacial dimensionality (width and height) of square image
        :param img_channels (int): number of channels for the input image
        :param latent_dim (int): size of latent vector to be used
        :param enc_conv_neurons (list): list denoting the number of output channels for each encoder convolutional layer
        :param dec_conv_neurons (list): list denoting the number of output channels for each decoder convolutional layer
        """

        super(VAE, self).__init__()

        if enc_conv_neurons is None:
            enc_conv_neurons = [32, 64, 128]
        if dec_conv_neurons is None:
            dec_conv_neurons = [128, 64]

        self.img_dim = img_dim
        self.img_channels = img_channels
        self.latent_dim = latent_dim
        self.enc_conv_neurons = enc_conv_neurons
        self.dec_conv_neurons = dec_conv_neurons

        self.setup_encoder_layers()
        self.setup_decoder_layers()

        self.nonlinearity = F.relu

    def setup_encoder_layers(self):
        """
        Setup the network layers for the encoder network
        """
        padding = calculate_padding(self.img_dim, self.img_dim, 5, 1)
        self.enc_conv1 = nn.Conv2d(in_channels=self.img_channels, out_channels=self.enc_conv_neurons[0], kernel_size=5, stride=1, padding=padding)
        self.enc_conv2 = nn.Conv2d(in_channels=self.enc_conv_neurons[0], out_channels=self.enc_conv_neurons[1], kernel_size=5, stride=1, padding=padding)
        self.enc_conv3 = nn.Conv2d(in_channels=self.enc_conv_neurons[1], out_channels=self.enc_conv_neurons[2], kernel_size=5, stride=1, padding=padding)

        self.enc_fc1 = nn.Linear(in_features=self.img_dim*self.img_dim*self.enc_conv_neurons[-1], out_features=128)
        self.enc_fc2_mean = nn.Linear(in_features=128, out_features=self.latent_dim)
        self.enc_fc2_logvar = nn.Linear(in_features=128, out_features=self.latent_dim)

    def setup_decoder_layers(self):
        """
        Setup the network layers for the decoder network
        """
        self.dec_fc1 = nn.Linear(in_features=self.latent_dim, out_features=128)
        self.dec_fc2 = nn.Linear(in_features=128, out_features=self.img_dim**2)

        padding = calculate_padding(self.img_dim, self.img_dim, 5, 1)
        self.dec_convT1 = nn.ConvTranspose2d(in_channels=1, out_channels=self.dec_conv_neurons[0], kernel_size=5, stride=1, padding=padding)
        self.dec_convT2 = nn.ConvTranspose2d(in_channels=self.dec_conv_neurons[0], out_channels=self.dec_conv_neurons[1], kernel_size=5, stride=1, padding=padding)
        self.dec_convT3 = nn.ConvTranspose2d(in_channels=self.dec_conv_neurons[1], out_channels=self.img_channels, kernel_size=5, stride=1, padding=padding)


    def encode(self, x):
        """
        Encoder portion of VAE
        :param x (tensor): Input to the network (batch ofimages)
        :return: mean(tensor), logvar(tensor): mean and log of variance for the multivariate gaussian distributin
                    parameterized by the encoder.
        """
        x = self.nonlinearity(self.enc_conv1(x))
        x = self.nonlinearity(self.enc_conv2(x))
        x = self.nonlinearity(self.enc_conv3(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.nonlinearity(self.enc_fc1(x))
        mean, logvar = self.enc_fc2_mean(x), self.enc_fc2_logvar(x)

        return mean, logvar

    def sample_latent(self, mean, logvar):
        """
        Sample the latent vector from the mean and logvar output by the encoder, using the parameterization trick
        :param mean: mean vector of MV Gaussian to sample from
        :param logvar: log of variance of MV Gaussian to sample from
        :return: latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)  # return z sample

    def decode(self, latent):
        """
        Decode the image from the latent vector
        :param latent: latent vector
        :return: image
        """
        batch_size = latent.shape[0]

        o = self.nonlinearity(self.dec_fc1(latent))
        o = self.nonlinearity(self.dec_fc2(o))
        o = o.view((batch_size, self.img_channels, self.img_dim, self.img_dim))
        o = self.nonlinearity(self.dec_convT1(o))
        o = self.nonlinearity(self.dec_convT2(o))
        o = F.sigmoid(self.dec_convT3(o))

        return o

    def forward(self, x):
        """
        Forward pass through VAE
        :param x: input
        :return: output, mean of encoder Gaussian parameterization, logvar of encoder GAussian parameterization
        """
        mean, logvar = self.encode(x)
        latent = self.sample_latent(mean, logvar)
        out = self.decode(latent)

        return out, mean, logvar

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def loss_function(x_pred, x, mean, logvar):
    """
    VAE loss function
    :param x_pred: VAE output
    :param x: input
    :param mean: mean output by VAE encoder
    :param logvar: logvar output by VAE encoder
    :return: loss for batch
    """
    recon_loss_fn = F.binary_cross_entropy
    reconstruction_loss = recon_loss_fn(x_pred, x, reduction='sum')
    KLD_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return reconstruction_loss + KLD_loss


def train_function(vae, optimizer, num_epochs, dataset, batch_size):
    dataset_size = dataset.shape[0]
    # shuffle dataset
    dataset = dataset[np.random.choice(dataset_size, dataset_size, replace=False)]
    for epoch in range(num_epochs):
        print("\n\n********EPOCH********: ", epoch, "\n\n")
        # get batch for minibatch SGD
        for batch_num in range(dataset_size // batch_size):
            inds = np.arange(batch_num*batch_size, (batch_num + 1)*batch_size)
            batch = dataset[inds]

            # indicate that this is training mode
            vae.train()
            # zero out gradients
            optimizer.zero_grad()

            # get prediction and latent state
            preds, means, log_vars = vae(batch)

            # loss and backprop
            loss = loss_function(preds, batch, means, log_vars)
            print("Loss at iter ", batch_num, " ", loss)
            loss.backward()
            optimizer.step()

def train_epoch(vae, optimizer, data_loader, epoch_num, batch_size):
    vae.train()
    epoch_loss = 0
    total_steps = 0
    for i, (batch, _) in enumerate(data_loader):
        optimizer.zero_grad()
        batch_pred, means, logvars = vae(batch)

        loss = loss_function(batch_pred, batch, means, logvars)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() / batch_size
        total_steps = i

        if total_steps % 20 == 0:
            print("Epoch:", epoch_num, " loss after", total_steps, "batches:", loss.item() / batch_size)

    return epoch_loss / total_steps

def train(dataset, vae, batch_size, num_epochs):
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(vae.parameters())

    for epoch in range(num_epochs):
        print("\n\n********EPOCH********: ", epoch)
        average_epoch_loss = train_epoch(vae, optimizer, data_loader, epoch, batch_size)
        print("Average loss for epoch: ", epoch, ": ", average_epoch_loss)


def generate_images(vae, num_images, latent_dim, save_path):
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim)
        sample = vae.decode(z)
        save_image(sample, save_path)

def main():
    # put save paths here
    MNIST_path = '/MNIST/'
    SAVE_PATH = '/VAE/vae.pt'
    IMG_SAVE_PATH = '/VAE/images.png'

    train_dataset = datasets.MNIST(root=MNIST_path, train=True, transform=transforms.ToTensor(),
                                   download=False)
    latent_dim = 50
    img_dim = 28
    num_channels = 1
    vae = VAE(img_dim, num_channels, latent_dim)

    num_epochs = 1
    batch_size = 100
    train(train_dataset, vae, batch_size, num_epochs)

    torch.save(vae.state_dict(), SAVE_PATH)

    generate_images(vae, 100, latent_dim, IMG_SAVE_PATH)


if __name__ == "__main__":
    with torch.autograd.detect_anomaly():
        main()