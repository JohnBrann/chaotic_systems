import numpy as np
import matplotlib.pyplot as plt

class JuliaSet:
    def __init__(self, c, width=800, height=800, max_iter=256, x_range=(-2, 2), y_range=(-2, 2)):
        self.c = c  # The complex constant that defines the Julia set
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.xmin, self.xmax = x_range
        self.ymin, self.ymax = y_range

    def generate(self):
        """Generates the Julia set based on the initialized parameters."""
        # Create a complex grid for the Julia set calculation
        x = np.linspace(self.xmin, self.xmax, self.width)
        y = np.linspace(self.ymin, self.ymax, self.height)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        
        # Initialize the output array
        output = np.zeros(Z.shape, dtype=int)

        # Iterate and calculate each point
        for i in range(self.max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask] ** 2 + self.c
            output += mask

        self.output = output

    def plot(self):
        """Plots the generated Julia set using matplotlib."""
        if not hasattr(self, 'output'):
            print("Generating the Julia set...")
            self.generate()
        
        plt.imshow(self.output.T, cmap='twilight', extent=[self.xmin, self.xmax, self.ymin, self.ymax])
        plt.colorbar()
        plt.title(f"Julia Set with c = {self.c}")
        plt.show()
    
    def generate_julia_data(c_values, width=80, height=80, max_iter=256):
        images = []
        constants = []

        for c in c_values:
            # Generate the Julia set
            x = np.linspace(-2, 2, width)
            y = np.linspace(-2, 2, height)
            X, Y = np.meshgrid(x, y)
            Z = X + 1j * Y
            
            output = np.zeros(Z.shape, dtype=int)

            for i in range(max_iter):
                mask = np.abs(Z) <= 2
                Z[mask] = Z[mask] ** 2 + c
                output += mask

            images.append(output)
            constants.append(c)

        return np.array(images), np.array(constants)

def main():
    # Create instances of JuliaSet with different constants
    julia1 = JuliaSet(c=-0.7+ 0.276j, max_iter=300)
    julia1.plot()

    julia2 = JuliaSet(c=-0.8 + 0.156j, max_iter=300)
    julia2.plot()

    julia3 = JuliaSet(c=0.355 + 0.355j, max_iter=400)
    julia3.plot()

if __name__ == "__main__":
    main()
