from torch import nn

class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        sample_dim: the dimension of the samples, fitted for the dataset used, a scalar.
        hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self, z_dim=10, sample_dim=1, hidden_dim=128):
        super(Generator, self).__init__()
        # Build the neural network: 3 layers, 1 output layer
        self.gen = nn.Sequential(
            nn.Linear(1, 7),
            nn.ReLU(inplace=True),
            nn.Linear(7, 13),
            nn.ReLU(inplace=True),
            nn.Linear(13, 7),
            nn.ReLU(inplace=True),
            nn.Linear(7, 1)
        )
    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated samples from the learnt distribution.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return self.gen(noise)

class GeneratorLeak(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        sample_dim: the dimension of the samples, fitted for the dataset used, a scalar.
        hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self, z_dim=10, sample_dim=1, hidden_dim=128):
        super(GeneratorLeak, self).__init__()
        # Build the neural network: 3 layers, 1 output layer
        self.gen = nn.Sequential(
            nn.Linear(1, 7),
            nn.LeakyReLU(inplace=True),
            nn.Linear(7, 13),
            nn.LeakyReLU(inplace=True),
            nn.Linear(13, 7),
            nn.LeakyReLU(inplace=True),
            nn.Linear(7, 1)
        )
    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated samples from the learnt distribution.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return self.gen(noise)


class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        sample_dim: the dimension of the sample, fitted for the dataset used, a scalar
        hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self, sample_dim=1, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(1, 7),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(7, 13),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(13, 7),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(7, 1)
        )

    def forward(self, sample):
        '''
        Function for completing a forward pass of the discriminator: Given an sample tensor,
        returns a 1-dimension tensor representing fake/real.
        '''
        return self.disc(sample)

class Generator2(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        sample_dim: the dimension of the samples, fitted for the dataset used, a scalar.
        hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self, z_dim=10, sample_dim=1, hidden_dim=128):
        super(Generator2, self).__init__()
        # Build the neural network: 3 layers, 1 output layer
        self.gen = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 248),
            nn.ReLU(inplace=True),
            nn.Linear(248, 248*2),
            nn.ReLU(inplace=True),
            nn.Linear(248 * 2,248 * 4),
            nn.ReLU(inplace=True),
            nn.Linear(248 * 4, 1)
        )
    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated samples from the learnt distribution.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return self.gen(noise)

class Discriminator2(nn.Module):
    '''
    Discriminator Class
    Values:
        sample_dim: the dimension of the sample, fitted for the dataset used, a scalar
        hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self, sample_dim=1, hidden_dim=128):
        super(Discriminator2, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(1, 128*4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128*4, 128*2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128*2, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 1)
        )

    def forward(self, sample):
        '''
        Function for completing a forward pass of the discriminator: Given an sample tensor,
        returns a 1-dimension tensor representing fake/real.
        '''
        return self.disc(sample)

class Generator3(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        sample_dim: the dimension of the samples, fitted for the dataset used, a scalar.
        hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self, z_dim=10, sample_dim=1, hidden_dim=128):
        super(Generator3, self).__init__()
        # Build the neural network: 3 layers, 1 output layer
        self.gen = nn.Sequential(
            nn.Linear(1, 7),
            nn.ELU(alpha=1,inplace=True),
            nn.Linear(7, 13),
            nn.ELU(alpha=1,inplace=True),
            nn.Linear(13, 7),
            nn.ELU(alpha=1,inplace=True),
            nn.Linear(7, 1)
        )

    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated samples from the learnt distribution.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return self.gen(noise)

class Discriminator3(nn.Module):
    '''
    Discriminator Class
    Values:
        sample_dim: the dimension of the sample, fitted for the dataset used, a scalar
        hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self, sample_dim=1, hidden_dim=128):
        super(Discriminator3, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(1, 11),
            nn.ELU(alpha=1, inplace=True),
            nn.Linear(11, 29),
            nn.ELU(alpha=1, inplace=True),
            nn.Linear(29, 11),
            nn.ELU(alpha=1, inplace=True),
            nn.Linear(11, 1),
        )

    def forward(self, sample):
        '''
        Function for completing a forward pass of the discriminator: Given an sample tensor,
        returns a 1-dimension tensor representing fake/real.
        '''
        return self.disc(sample)

class Generator_z(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        sample_dim: the dimension of the samples, fitted for the dataset used, a scalar.
        hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self, z_dim=10, sample_dim=1, hidden_dim=128):
        super(Generator_z, self).__init__()
        # Build the neural network: 3 layers, 1 output layer
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 7),
            nn.ReLU(inplace=True),
            nn.Linear(7, 13),
            nn.ReLU(inplace=True),
            nn.Linear(13, 7),
            nn.ReLU(inplace=True),
            nn.Linear(7, 1)
        )
    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated samples from the learnt distribution.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return self.gen(noise)

class Generator_z2(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        sample_dim: the dimension of the samples, fitted for the dataset used, a scalar.
        hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self, z_dim=20, sample_dim=1, hidden_dim=128):
        super(Generator_z2, self).__init__()
        # Build the neural network: 3 layers, 1 output layer
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated samples from the learnt distribution.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return self.gen(noise)

class Generator_Lz2(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        sample_dim: the dimension of the samples, fitted for the dataset used, a scalar.
        hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self, z_dim=20, sample_dim=1, hidden_dim=128):
        super(Generator_Lz2, self).__init__()
        # Build the neural network: 3 layers, 1 output layer
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 1)
        )
    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated samples from the learnt distribution.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return self.gen(noise)

class Discriminator_z2(nn.Module):
    '''
    Discriminator Class
    Values:
        sample_dim: the dimension of the sample, fitted for the dataset used, a scalar
        hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self, sample_dim=1, hidden_dim=128):
        super(Discriminator_z2, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(1, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 1)
        )

    def forward(self, sample):
        '''
        Function for completing a forward pass of the discriminator: Given an sample tensor,
        returns a 1-dimension tensor representing fake/real.
        '''
        return self.disc(sample)

# MMD GAN
class GNet(nn.Module):
    def __init__(self):
        super().__init__()
        model = nn.Sequential(
            nn.Linear(1, 7),
            nn.ELU(),
            nn.Linear(7, 13),
            nn.ELU(),
            nn.Linear(13, 7),
            nn.ELU(),
            nn.Linear(7, 1)
        )
        self.model = model
    def forward(self, input):
        return self.model(input)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        model = nn.Sequential(
            nn.Linear(1, 11),
            nn.ELU(),
            nn.Linear(11, 29),
            nn.ELU()
        )
        self.model = model
    def forward(self, input):
        return self.model(input)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        model = nn.Sequential(
            nn.Linear(29, 11),
            nn.ELU(),
            nn.Linear(11, 1),
        )
        self.model = model
    def forward(self, input):
        return self.model(input)

