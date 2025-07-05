# Import necessary libraries
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import os

# --- Configuration and Hyperparameters ---
BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 5000 
NOISE_DIM = 100
NUM_EXAMPLES_TO_GENERATE = 16

class GenerativeAdversarialNetwork:
    """ Encapsulates the GAN models and their core logic. """
    def __init__(self):
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])

    def build_generator(self):
        """Builds the larger and more capable Generator model."""
        model = tf.keras.Sequential(name="Generator")
        model.add(layers.Dense(7*7*512, use_bias=False, input_shape=(NOISE_DIM,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Reshape((7, 7, 512)))
        model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
        return model

    def build_discriminator(self):
        """Builds the larger and more capable Discriminator model."""
        model = tf.keras.Sequential(name="Discriminator")
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.3))
        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.3))
        model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.3))
        model.add(layers.Flatten())
        model.add(layers.Dense(1))
        return model

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def generate_images(self, noise_seed=None):
        if noise_seed is None:
            noise_seed = self.seed
        predictions = self.generator(noise_seed, training=False)
        return predictions.numpy()

    def save_models(self, path):
        self.generator.save_weights(os.path.join(path, 'generator.h5'))
        self.discriminator.save_weights(os.path.join(path, 'discriminator.h5'))

    def load_models(self, path):
        self.generator.load_weights(os.path.join(path, 'generator.h5'))
        self.discriminator.load_weights(os.path.join(path, 'discriminator.h5'))


class GAN_GUI:
    """ The main application class for the Tkinter GUI. """
    def __init__(self, master):
        self.master = master
        master.title("Interactive Generative AI")
        master.geometry("850x800")

        # --- State Variables ---
        self.is_training = False
        self.training_paused = True
        self.epoch = 0
        self.global_step = 0
        self.view_mode = 'grid'
        self.latent_start_vector = None
        self.latent_end_vector = None
        self.is_tripping = False # For automatic animation
        self.trip_thread = None

        # --- Setup GAN and Data ---
        self.gan = GenerativeAdversarialNetwork()
        self.train_dataset = self.load_dataset()

        # --- Create GUI Widgets ---
        self.create_widgets()
        self.update_image_display() 

    def load_dataset(self):
        (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = (train_images - 127.5) / 127.5
        return tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    def create_widgets(self):
        """Creates and arranges all the GUI elements."""
        # --- Main Frames ---
        top_frame = tk.Frame(self.master)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        training_frame = ttk.LabelFrame(top_frame, text="Training Controls")
        training_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5), ipady=5)

        latent_frame = ttk.LabelFrame(top_frame, text="Latent Space Explorer")
        latent_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0), ipady=5)

        # --- **NEW** Effects Frame ---
        effects_frame = ttk.LabelFrame(self.master, text="Animation & Effects ('Drugs')")
        effects_frame.pack(fill=tk.X, padx=10, pady=5, ipady=5)

        # --- Image Display ---
        self.fig = plt.figure(figsize=(6, 6))
        self.fig.patch.set_facecolor('#F0F0F0')
        image_frame = tk.Frame(self.master)
        image_frame.pack(pady=10, expand=True)
        self.image_canvas = FigureCanvasTkAgg(self.fig, master=image_frame)
        self.image_canvas.get_tk_widget().pack()

        # --- Status Bar ---
        status_frame = tk.Frame(self.master, pady=5)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label = tk.Label(status_frame, text="Status: Idle | Epoch: 0")
        self.status_label.pack()

        # --- Training Control Widgets ---
        self.train_button = tk.Button(training_frame, text="Start Training", command=self.start_training)
        self.train_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.pause_button = tk.Button(training_frame, text="Pause Training", command=self.pause_training, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.save_button = tk.Button(training_frame, text="Save Models", command=self.save_models)
        self.save_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.load_button = tk.Button(training_frame, text="Load Models", command=self.load_models)
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # --- Latent Space Widgets ---
        self.grid_button = tk.Button(latent_frame, text="Generate Grid", command=self.set_grid_view)
        self.grid_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.set_start_button = tk.Button(latent_frame, text="Set Start Image", command=self.set_latent_start)
        self.set_start_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.set_end_button = tk.Button(latent_frame, text="Set End Image", command=self.set_latent_end)
        self.set_end_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.morph_slider = tk.Scale(latent_frame, from_=0, to=100, orient=tk.HORIZONTAL, length=200, label="Morph", command=self.update_interpolation_view)
        self.morph_slider.pack(side=tk.LEFT, padx=5, pady=5)

        # --- **NEW** Effects Widgets ---
        self.trip_button = tk.Button(effects_frame, text="Start Trip", command=self.toggle_trip)
        self.trip_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.noise_slider = tk.Scale(effects_frame, from_=0, to=100, orient=tk.HORIZONTAL, length=200, label="Noise Injection", command=self.update_interpolation_view)
        self.noise_slider.pack(side=tk.LEFT, padx=5, pady=5)
        self.save_art_button = tk.Button(effects_frame, text="Save Current Image", command=self.save_current_image)
        self.save_art_button.pack(side=tk.LEFT, padx=5, pady=5)

    def set_grid_view(self):
        self.stop_trip()
        self.view_mode = 'grid'
        self.gan.seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])
        self.update_image_display()

    def set_latent_start(self):
        self.stop_trip()
        self.view_mode = 'single'
        self.latent_start_vector = tf.random.normal([1, NOISE_DIM])
        self.update_image_display(noise_seed=self.latent_start_vector)
        self.morph_slider.set(0)

    def set_latent_end(self):
        self.stop_trip()
        self.view_mode = 'single'
        self.latent_end_vector = tf.random.normal([1, NOISE_DIM])
        self.update_image_display(noise_seed=self.latent_end_vector)
        self.morph_slider.set(100)

    def update_interpolation_view(self, val=None):
        if self.latent_start_vector is None or self.latent_end_vector is None:
            return
        
        alpha = self.morph_slider.get() / 100.0
        interpolated_noise = self.slerp(alpha, self.latent_start_vector, self.latent_end_vector)
        
        # **NEW**: Add noise injection
        noise_level = self.noise_slider.get() / 100.0
        if noise_level > 0:
            random_noise = tf.random.normal(shape=interpolated_noise.shape) * noise_level
            interpolated_noise += random_noise

        self.view_mode = 'single'
        self.update_image_display(noise_seed=interpolated_noise)

    def slerp(self, val, low, high):
        omega = tf.acos(tf.clip_by_value(tf.reduce_sum(low/tf.norm(low) * high/tf.norm(high)), -1, 1))
        so = tf.sin(omega)
        if so == 0:
            return (1.0-val) * low + val * high
        return (tf.sin((1.0-val)*omega)/so) * low + (tf.sin(val*omega)/so) * high

    def update_image_display(self, noise_seed=None):
        self.fig.clear()
        if self.view_mode == 'grid':
            axes = self.fig.subplots(4, 4)
            predictions = self.gan.generate_images(noise_seed=self.gan.seed)
            for i, ax in enumerate(axes.flat):
                ax.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
                ax.axis('off')
        else:
            ax = self.fig.add_subplot(1, 1, 1)
            if noise_seed is None:
                noise_seed = tf.random.normal([1, NOISE_DIM])
            prediction = self.gan.generate_images(noise_seed=noise_seed)
            ax.imshow(prediction[0, :, :, 0] * 127.5 + 127.5, cmap='gray')
            ax.axis('off')
        plt.tight_layout(pad=0.5)
        self.image_canvas.draw()

    # --- **NEW** Feature Methods ---
    def toggle_trip(self):
        if self.is_tripping:
            self.stop_trip()
        else:
            self.start_trip()

    def start_trip(self):
        if self.is_training:
            messagebox.showwarning("Warning", "Pause training before starting a trip.")
            return
        self.is_tripping = True
        self.trip_button.config(text="Stop Trip")
        self.trip_thread = threading.Thread(target=self._trip_loop, daemon=True)
        self.trip_thread.start()

    def stop_trip(self):
        self.is_tripping = False
        if self.trip_thread:
            self.trip_thread.join(timeout=0.1)
        self.trip_button.config(text="Start Trip")

    def _trip_loop(self):
        """The core animation loop for the 'trip' feature."""
        # Ensure we have start/end points
        if self.latent_start_vector is None:
            self.latent_start_vector = tf.random.normal([1, NOISE_DIM])
        self.latent_end_vector = tf.random.normal([1, NOISE_DIM])
        
        while self.is_tripping:
            # Animate from 0 to 100
            for i in range(101):
                if not self.is_tripping: break
                self.morph_slider.set(i)
                self.master.after(0, self.update_interpolation_view)
                time.sleep(0.03)
            
            if not self.is_tripping: break
            time.sleep(0.5) # Pause at the end

            # Animate from 100 to 0
            for i in range(100, -1, -1):
                if not self.is_tripping: break
                self.morph_slider.set(i)
                self.master.after(0, self.update_interpolation_view)
                time.sleep(0.03)

            # Set up the next leg of the trip
            self.latent_start_vector = self.latent_end_vector
            self.latent_end_vector = tf.random.normal([1, NOISE_DIM])
        
        # Ensure button is reset on the main thread
        self.master.after(0, self.trip_button.config, {'text': 'Start Trip'})

    def save_current_image(self):
        """Saves the content of the matplotlib canvas to a file."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            title="Save Image As"
        )
        if not file_path:
            return
        try:
            self.fig.savefig(file_path, dpi=300, facecolor='#F0F0F0')
            messagebox.showinfo("Success", f"Image saved successfully to\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {e}")

    # --- Training Functions ---
    def start_training(self):
        self.stop_trip()
        if not self.is_training:
            self.is_training = True
            self.training_paused = False
            self.train_thread = threading.Thread(target=self._training_loop, daemon=True)
            self.train_thread.start()
            self.train_button.config(text="Resume Training")
        else:
            self.training_paused = False
        self.train_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.load_button.config(state=tk.DISABLED)

    def pause_training(self):
        self.training_paused = True
        self.train_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.load_button.config(state=tk.NORMAL)
        self.update_status("Paused")

    def _training_loop(self):
        dataset_iterator = iter(self.train_dataset.repeat())
        while self.epoch < EPOCHS and self.is_training:
            if not self.training_paused:
                self.update_status("Training...")
                image_batch = next(dataset_iterator)
                self.gan.train_step(image_batch)
                if self.global_step > 0 and self.global_step % (BUFFER_SIZE // BATCH_SIZE) == 0:
                    self.epoch += 1
                    if self.view_mode == 'grid':
                        self.master.after(0, self.update_image_display)
                self.global_step += 1
                self.update_status(f"Training... Epoch {self.epoch}")
            time.sleep(0.01)
        self.is_training = False
        self.update_status("Training Finished.")
        self.master.after(0, self.pause_training)

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message} | Epoch: {self.epoch}")

    def save_models(self):
        path = filedialog.askdirectory(title="Select Folder to Save Models")
        if path:
            try:
                self.gan.save_models(path)
                messagebox.showinfo("Success", f"Models saved successfully in {path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save models: {e}")

    def load_models(self):
        path = filedialog.askdirectory(title="Select Folder with Saved Models")
        if path:
            try:
                self.stop_trip()
                self.gan.load_models(path)
                self.set_grid_view()
                messagebox.showinfo("Success", f"Models loaded successfully from {path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load models: {e}\nEnsure generator.h5 and discriminator.h5 are in the folder.")


if __name__ == '__main__':
    root = tk.Tk()
    app = GAN_GUI(root)
    root.mainloop()
