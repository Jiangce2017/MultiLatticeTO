import numpy as np
import pacmap  # will need to change numba version: pip install numba==0.53
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from smoothness_testing import smoothness
import cv2
from matplotlib.collections import LineCollection

########################################################################################################################
# Latent Feature Cluster for Training Data using PaCMAP
def PaCMAP_reduction(latent_points, random_state=1):
    # initializing the pacmap instance
    X = latent_points
    latent_dimensionality = len(latent_points[0])
    embedding = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=random_state)

    # fit the data (The index of transformed data corresponds to the index of the original data)
    X_transformed = embedding.fit_transform(X, init="pca")

    # visualize the embedding
    x = X_transformed[:, 0]
    y = X_transformed[:, 1]
    title = "PaCMAP with Predicted Points\nLatent Space Dimensionality: " + str(latent_dimensionality)
    return x, y, title, embedding


########################################################################################################################
# Latent Feature Cluster for Training Data using PCA and Predicted Latent Points
def PCA_reduction(latent_points):
    latent_dimensionality = len(latent_points[0])
    pca = PCA(n_components=2, random_state=0)
    embedding = pca
    pca_fit = pca.fit_transform(latent_points)
    # configuring the parameters
    # the number of components = dimension of the embedded space
    # default perplexity = 30 " Perplexity balances the attention t-SNE gives to local and global aspects of the data.
    # It is roughly a guess of the number of close neighbors each point has. ..a denser dataset ... requires higher perplexity value"
    # default learning rate = 200 "If the learning rate is too high, the data may look like a ‘ball’ with any point
    # approximately equidistant from its nearest neighbours. If the learning rate is too low,
    # most points may look compressed in a dense cloud with few outliers."
    title = "PCA with Predicted Points\nLatent Space Dimensionality: " + str(latent_dimensionality)
    x = pca_fit[:, 0]
    y = pca_fit[:, 1]

    return x, y, title, embedding


########################################################################################################################
# Latent Feature Cluster for Training Data using T-SNE
def TSNE_reduction(latent_points, perplexity=30, learning_rate=20):
    latent_dimensionality = len(latent_points[0])
    model = TSNE(n_components=2, random_state=0, perplexity=perplexity,
                 learning_rate=learning_rate)  # Perplexity(5-50) | learning_rate(10-1000)
    embedding = model
    # configuring the parameters
    # the number of components = dimension of the embedded space
    # default perplexity = 30 " Perplexity balances the attention t-SNE gives to local and global aspects of the data.
    # It is roughly a guess of the number of close neighbors each point has. ..a denser dataset ... requires higher perplexity value"
    # default learning rate = 200 "If the learning rate is too high, the data may look like a ‘ball’ with any point
    # approximately equidistant from its nearest neighbours. If the learning rate is too low,
    # most points may look compressed in a dense cloud with few outliers."
    tsne_data = model.fit_transform(
        latent_points)  # When there are more data points, trainX should be the first couple hundred points so TSNE doesn't take too long
    x = tsne_data[:, 0]
    y = tsne_data[:, 1]
    title = ("T-SNE of Data\nPerplexity: " + str(perplexity) + "\nLearning Rate: "
             + str(learning_rate) + "\nLatent Space Dimensionality: " + str(latent_dimensionality))
    return x, y, title, embedding


########################################################################################################################
# Generate Embeddings to be used in latent dimensionality reduction applications
def generate_embeddings(train_dataset_latent_points, embedding_type='all', perplexity=30, learning_rate=20):
    """
    :param train_dataset_latent_points: - the latent points of a given dataset, should use training data for embeddings
    :param perplexity: - Affects the performance of the TSNE. Perplexity should be roughly the sqrt(N) where N is the
    number of samples.
    :param learning_rate: - Afffects the performance of TSNE: "If the learning rate is too high, the data may look like a
    ‘ball’ with any point approximately equidistant from its nearest neighbours. If the learning rate is too low,
    most points may look compressed in a dense cloud with few outliers."
    :param embedding_type: - the type of embedding desired for the plot. 'pca', 'tsne' or 'pacmap'
    :return: pca, PaCMAP, TSNE - returns the pca embedding which can be used to embed test points for plotting
    interpolations. Currently, the PaCMAP and TSNE embeddings do not work, need to define a new function for using them,
    as they require the test points to be embedded as well.
    """
    if embedding_type=='all':
        pca = PCA_reduction(train_dataset_latent_points)  # x, y, title, embedding
        PaCMAP = PaCMAP_reduction(train_dataset_latent_points, random_state=1)  # x, y, title, embedding
        TSNE = TSNE_reduction(train_dataset_latent_points, perplexity=perplexity,
                              learning_rate=learning_rate)  # x, y, title, embedding
        return pca, PaCMAP, TSNE
    elif embedding_type == "pca":
        pca = PCA_reduction(train_dataset_latent_points)  # x, y, title, embedding
        return pca
    elif embedding_type == 'tsne':
        TSNE = TSNE_reduction(train_dataset_latent_points, perplexity=perplexity,
                              learning_rate=learning_rate)  # x, y, title, embedding
        return TSNE
    elif embedding_type == 'pacmap':
        PaCMAP = PaCMAP_reduction(train_dataset_latent_points, random_state=1)  # x, y, title, embedding
        return PaCMAP

########################################################################################################################
"""
def plot_dimensionality_reduction(x, y, label_set, title):
    plt.title(title)
    if label_set[0].dtype == float:
        plt.scatter(x, y, c=label_set)
        plt.colorbar()
        print("using scatter")
    else:
        for label in set(label_set):
            cond = np.where(np.array(label_set) == str(label))
            plt.plot(x[cond], y[cond], marker='o', linestyle='none', label=label)

        plt.legend(numpoints=1)

    plt.show()
    plt.close()
"""


def plot_dimensionality_reduction(x, y, label_set, title,fig=None, ax=None, color_mapping='viridis'):
    if fig is None and ax is None:
        fig, ax = plt.subplots()
        plt.title(title)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
    # Color points based on their density
    cmap = matplotlib.colormaps[color_mapping]
    if label_set[0].dtype == float:
        scatter = ax.scatter(x, y, c=label_set, cmap=cmap)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Average Density', fontsize=12)
        print("using scatter")


    # Color points based on a discrete label
    else:
        for label in set(label_set):
            cond = np.where(np.array(label_set) == str(label))
            ax.plot(x[cond], y[cond], marker='o', linestyle='none', label=label)

        plt.legend(numpoints=1)



########################################################################################################################
def plot_dimensionality_reduction_average_density(data, embedding, type_of_reduction, fig=None, ax=None):
    # use training data (use geometry)
    fig_ax = False
    if fig is None and ax is None:
        fig, ax = plt.subplots()
        fig_ax = True  # Define a variable to

    number_samples = len(data)

    def density_calc(matrix):
        density = np.sum(matrix) / len(matrix) ** 2
        return density

    avg_density = np.array(list(map(density_calc, data)))
    x, y, title, reduction_embedding = embedding
    plot_dimensionality_reduction(x, y, avg_density, title, fig=fig, ax=ax)

    if fig_ax:
        plt.title(title)
        plt.savefig(type_of_reduction + "_Dimensionality_Reduction" + "_Data_Samples_" + str(number_samples), bbox_inches='tight')
        plt.close()


########################################################################################################################
def plot_smoothness_of_interps_with_reduction(data, embedding, interpolated_latent_points, smoothness_values_interpolation, type_of_reduction, file_name, color_bar_min=85):
    # use training data

    # arrays = pd.DataFrame(box_matrix_train)
    x, y, title, reduction_embedding = embedding

    # Use the Embedding to append points for the Interpolated Images
    embedded_interpolated_latent_points = reduction_embedding.transform(interpolated_latent_points)

    fig, ax = plt.subplots()

    # Create the Segments between the rows and columns in the Mesh
    segments = []
    for i in range(np.shape(embedded_interpolated_latent_points)[0]-1):
        segments.append([embedded_interpolated_latent_points[i], embedded_interpolated_latent_points[i+1]])

    plt.scatter(embedded_interpolated_latent_points[:,0], embedded_interpolated_latent_points[:,1], c='r', zorder=25)

    # Normalize the smoothness between 0 and 1
    smoothness_values_interpolation = np.array(smoothness_values_interpolation) / 100  # Calculates the smoothness of each row
    # Plot the Line segments of the interpolation

    plot_line_segments(segments, smoothness_values_interpolation, ax, color_bar_min=color_bar_min, color_bar_max=100)

    # Plot the points of the latent space
    def density_calc(matrix):
        density = np.sum(matrix) / len(matrix) ** 2
        return density

    avg_density = np.array(list(map(density_calc, data)))

    plot_dimensionality_reduction(x, y, avg_density, title, color_mapping='Greys', fig=fig, ax=ax)
    plt.title(title)

    plt.savefig(type_of_reduction + file_name, bbox_inches='tight')
    plt.show()
    plt.close()


########################################################################################################################
# Scatter with images instead of points
def imscatter(x, y, ax, imageData, zoom):
    images = []
    for i in range(len(x)):
        x0, y0 = x[i], y[i]
        image_size = np.shape(imageData[0])
        # Convert to image
        img = imageData[i] * 255.
        img = img.astype(np.uint8).reshape([image_size[0], image_size[1]])
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # Note: OpenCV uses BGR and plt uses RGB
        image = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
        images.append(ax.add_artist(ab))

    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()


# Plot images in latent space with respective reduction method
def Latent_Image_Proj(image_arrays, embedding):
    # Compute Reduction embedding of latent space
    x, y, title, reduction_embedding = embedding
    # Plot images according to reduction embedding
    image_arrays = np.pad(image_arrays, 1, mode='constant')
    fig, ax = plt.subplots()
    imscatter(x, y, imageData=image_arrays, ax=ax, zoom=0.6)
    plt.title(title)
    plt.show()


########################################################################################################################
def plot_interpolation_smoothness(original_data_labels, train_dataset_latent_points, interpolated_latent_points, embedding_type, train_image_arrays,
                                  image_size, number_of_interpolations, markersize=8,
                                  marker_color='black', mesh_predicted_interps=None, plot_train_images=True,
                                  plot_points=True, color_bar_min=85, color_bar_max=100, title="",
                                  plot_row_segments=False, plot_col_segments=False, plot_lines=False):
    """
    :param original_data_labels: - labels to define the latent space data set, can be arbitrary, just needs to provide
    differentiation from the predicted points. Needs to match the length of the train_dataset_latent_points parameter.
    :param train_dataset_latent_points: - the latent points for the space that is to define the embedding of the data.
    Typically, a training dataset, as it is much larger and more representative of the data.
    :param interpolated_latent_points: - the latent points of each of the predicted data points in the interpolation.
    These points need to be in a 1-D list.
    :param embedding_type: - the type of embedding desired for the plot. 'pca', 'tsne' or 'pacmap'
    :param train_image_arrays: - the training data images to use for plotting the images in the embedding.
    :param image_size: -
    :param number_of_interpolations: -
    :param markersize: - Determines the marker size for the predicted interpolation points.
    :param marker_color: - Determines the marker color for the predicted interpolation points.
    :param mesh_predicted_interps: - if the interpolation is 2D (a mesh) then the images of the points need to be
    provided in order to choose to display the corner images. Creates additional plots of the smoothness over each row
    and column of the mesh.
    :param plot_train_images: - if True, the training images will be plotted based on the mesh or 1D interpolation.
    If False, then the plot will display the training points as an average density point.
    :param plot_points: - plot the points of the mesh or not
    :param color_bar_min: - the minimum of the colorbar for the smoothness values
    :param color_bar_max: - the maximum of the colorbar for the smoothness values
    :param title: -
    :param plot_row_segments: - plot the smoothness of the rows in the mesh
    :param plot_col_segments: - plot the smoothness of the columns in the mesh
    :param plot_lines: - used to plot 1-D interpolation, a feature that displays all the lines connecting consecutive
    points.
    :return: Plots - generates plots based on the parameters specified
    """
    # This function plots the lines of smoothness over the latent space of grayed out images, with the endpoints marked on a mesh
    # combines all the latent points of the training data and the interpolation
    # train_data_latent_points = np.append(original_data_latent_points, interpolated_latent_points, axis=0)

    # Flattens the mesh into a shape of [total number of samples, latent dimensions]
    latent_dimensionality = len(interpolated_latent_points[0][0])  # Returns the size of the first point
    interpolated_latent_points = np.reshape(interpolated_latent_points, (number_of_interpolations ** 2, latent_dimensionality))

    # Append the list of the interpolated points to the training data to be embedded
    combined_data_latent_points = np.concatenate((train_dataset_latent_points, interpolated_latent_points), axis=0)

    embedding = generate_embeddings(combined_data_latent_points, embedding_type=embedding_type)
    x1, y1, title1, reduction_embedding = embedding

    '''
    # Perform Reduction to get points for Training Images
    x1, y1, title1, reduction_embedding = embedding

    # Use the Embedding to append points for the Interpolated Images
    embedded_interpolated_latent_points = reduction_embedding.transform(interpolated_latent_points)
    x2 = embedded_interpolated_latent_points[:, 0]
    y2 = embedded_interpolated_latent_points[:, 1]

    x1 = np.append(x1, x2)
    y1 = np.append(y1, y2)
    '''

    # Check if there is an input for mesh_predicted_interps
    if mesh_predicted_interps is not None:
        print(np.shape(train_image_arrays))
        print(np.shape(mesh_predicted_interps))
        combined_data_images = np.concatenate((train_image_arrays, mesh_predicted_interps), axis=0)

        # reshape so that the images can be indexed by row/column
        # mesh_predicted_interps_flattened = np.reshape(mesh_predicted_interps.copy(), (number_of_interpolations ** 2,
        #                                                              image_size, image_size))
        mesh_predicted_interps = np.reshape(mesh_predicted_interps, (number_of_interpolations, number_of_interpolations, image_size, image_size))



        # Information needed to plot the smoothness of the rows and columns in the mesh
        # Get the smoothness of each row in the mesh
        count_row = []
        smoothness_line_row = []
        for row in range(np.shape(mesh_predicted_interps)[0]):
            count_row.append(row)
            interpolation = mesh_predicted_interps[row, :]
            smoothness_line_row.append(smoothness(interpolation)[0]) # adds the average smoothness to our array
        plt.scatter(count_row, smoothness_line_row, label="Row Smoothness")

        # Get the smoothness for each column in the mesh
        count_col = []
        smoothness_line_col = []
        for col in range(np.shape(mesh_predicted_interps)[1]):
            count_col.append(col)
            interpolation = mesh_predicted_interps[:, col]
            smoothness_line_col.append(smoothness(interpolation)[0])  # adds the average smoothness to our array
        plt.scatter(count_col, smoothness_line_col, label="Column Smoothness")

        plt.legend(fontsize=20)
        plt.xlabel("Rows/Columns", fontsize=20)
        plt.ylabel("Smoothness (%)", fontsize=20)
        plt.title("Smoothness over mesh ", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim([60, 100])
        plt.show()



    # Get labels for all the points in a single list
    combined_label = original_data_labels  # Contains the labels for all the points
    for i in range(len(interpolated_latent_points)):
        combined_label = np.append(combined_label, np.array("Predicted Points"))

    # Establish plot reduction of images
    image_arrays_padded = np.pad(train_image_arrays, 1, mode='constant')  # Puts a black box surrounding each array
    fig, ax = plt.subplots()

    # Sort and plot the points and images into the latent space
    for label in set(combined_label):
        cond = np.where(np.array(combined_label) == str(label))
        # Plotting for the training points
        if label != "Predicted Points":
            # Plot the training data as images
            if plot_train_images:
                # there is a mesh interpolation, then the background images will be gray boxes
                if mesh_predicted_interps is not None:
                    image_arrays = np.array(train_image_arrays)
                    image_arrays[image_arrays < 2] = 0.5  # Replaces the training images with gray boxes
                    image_arrays_gray = np.pad(image_arrays, 1, mode='constant')  # Puts a black box surrounding each array
                    images = image_arrays_gray
                # If there is a linear interpolation, then the background images will display their true values
                else:
                    images = image_arrays_padded
                # Plot the training images
                imscatter(x1[cond], y1[cond], imageData=images[cond], ax=ax, zoom=0.6) # , image_size=image_size + 2
            # Plot the training data as points with their density values
            elif mesh_predicted_interps is not None:
                plot_dimensionality_reduction_average_density(combined_data_images, embedding, "", fig=fig, ax=ax)


        # Plotting for the predicted points
        else:
            if plot_points is True:  # Plots the predicted points
                ax.plot(x1[cond], y1[cond], marker='o', c=marker_color, markersize=markersize, linestyle='none',
                        label=label, zorder=5)
            if plot_lines:
                ax.plot(x1[cond], y1[cond], 'ro-', zorder=10)

    # Perform Mesh Operations
    line_segment_title = ""
    if mesh_predicted_interps is not None:
        # Pull Coordinates from Reduction for plotting the Mesh
        interpolation_cords_x = x1[-np.shape(interpolated_latent_points)[0]:]  # coordinates of the interpolation points x(ordered)
        interpolation_cords_x = np.reshape(interpolation_cords_x, (np.shape(mesh_predicted_interps)[0], np.shape(mesh_predicted_interps)[1]))

        interpolation_cords_y = y1[-np.shape(interpolated_latent_points)[0]:]  # coordinates of the interpolation points y(ordered)
        interpolation_cords_y = np.reshape(interpolation_cords_y, (np.shape(mesh_predicted_interps)[0], np.shape(mesh_predicted_interps)[1]))

        # Create the Segments between the rows and columns in the Mesh
        row_lines = []
        for row in range(np.shape(interpolation_cords_x)[0]):
            row_lines.append([(interpolation_cords_x[row, 0], interpolation_cords_y[row, 0]),
                              (interpolation_cords_x[row, -1], interpolation_cords_y[row,-1])])
        col_lines = []
        for col in range(np.shape(interpolation_cords_x)[1]):
            col_lines.append([(interpolation_cords_x[0, col], interpolation_cords_y[0, col]),
                              (interpolation_cords_x[-1, col], interpolation_cords_y[-1, col])])

        # Plot the Line Segments in the rows and columns in the mesh
        smoothness_line_row = np.array(smoothness_line_row) / 100  # Calculates the smoothness of each row
        smoothness_line_col = np.array(smoothness_line_col) / 100  # Calculates the smoothness of each column

        if plot_row_segments == plot_col_segments == True:  # Plots rows and columns
            plot_line_segments_rows_columns(row_lines, col_lines, smoothness_line_row, smoothness_line_col, ax, "Row", "Column")
            line_segment_title = ": Smoothness of Rows and Columns Represented by Line Segments"

        elif plot_col_segments is True:  # Plots the columns only
            plot_line_segments(col_lines, smoothness_line_col, ax, color_bar_min=color_bar_min,
                               color_bar_max=color_bar_max)  # function that plots the line segments and color codes them
            line_segment_title = ": Smoothness Columns Represented by Line Segments"

        elif plot_row_segments is True:  # Plots the rows only
            plot_line_segments(row_lines, smoothness_line_row, ax, color_bar_min=color_bar_min,
                               color_bar_max=color_bar_max)  # function that plots the line segments and color codes them
            line_segment_title = ": Smoothness of Rows Represented by Line Segments"


        # Plotting the Images in the 4 Corners of the Mesh
        images_corners = []
        x_corners = []
        y_corners = []
        for point in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:  # Loop through the corner points in the mesh
            images_corners.append(np.pad(mesh_predicted_interps[point], 1, mode='constant')) # Puts a black box surrounding each array
            x_corners.append(interpolation_cords_x[point])
            y_corners.append(interpolation_cords_y[point])
        imscatter(x_corners, y_corners, imageData=images_corners, ax=ax, zoom=1.5) #, image_size=image_size + 2)

        # Plots the predicted points, line segments, training images, and images from the 4 corners of the mesh on a
        # single figure
    plt.legend(numpoints=1, fontsize=20)
    plt.title(title + line_segment_title)
    plt.show()



########################################################################################################################
def plot_line_segments(segments, smoothness_of_segment, ax, color_bar_min=85, color_bar_max=100, color_mapping='viridis'):
    # Segments - list of line coordinates
    # Smoothness of Segment - the smoothness of the images over the segment
    # ax - the predefined axis that is being used to plot the data

    # Setup Colorbar Color, Min and Max
    cmap = matplotlib.colormaps[color_mapping]  # A function that returns the color value of a number (0-1)
    norm = matplotlib.colors.Normalize(vmin=color_bar_min / 100,
                                       vmax=color_bar_max / 100)  # A function to normalize values between a desired min and max

    # Plot the Line segments
    line_segment_rows = LineCollection(segments, colors=cmap(norm(smoothness_of_segment)), linestyles='solid',
                                       zorder=20, linewidths=4)
    ax.add_collection(line_segment_rows)
    fig = plt.gcf()

    # Color bar settings for Line Segments
    cbar = fig.colorbar(line_segment_rows,
                        ticks=[0, norm(min(smoothness_of_segment)),norm(max(smoothness_of_segment)), 1])  # Locations of labels on Color Bar
    cbar.set_label('Smoothness (%)', fontsize=20)  # Title of the color bar
    cbar.ax.set_yticklabels(
        [str(color_bar_min),
         str(round(min(smoothness_of_segment) * 100, 2)) + " - Min",
         str(round(max(smoothness_of_segment) * 100, 2)) + " - Max",
         '100'], fontsize=16)  # Labels on Color Bar
    ax.autoscale()


########################################################################################################################
def plot_line_segments_rows_columns(segments1, segments2, smoothness_of_segment1, smoothness_of_segment2, ax,
                                    name_segment_1="Segment Set 1", name_segment_2="Segment Set 2",
                                    color_bar_min=85, color_bar_max=100):
    # Used to plot two different sets of segments on the same scale
    # Segments - list of line coordinates
    # Smoothness of Segment - the smoothness of the images over the segment
    # ax - the predefined axis that is being used to plot the data

    # Setup Colorbar Color, Min and Max
    cmap = matplotlib.colormaps['viridis']  # A function that returns the color value of a number (0-1)
    norm = matplotlib.colors.Normalize(vmin=color_bar_min / 100,
                                       vmax=color_bar_max / 100)  # A function to normalize values between a desired min and max
    # Combine Segments for Plotting
    collective_segments = np.append(segments1, segments2,axis=0)
    collective_smoothness = np.append(smoothness_of_segment1,smoothness_of_segment2)

    # Plot the Line segments
    line_segment_rows = LineCollection(collective_segments, colors=cmap(norm(collective_smoothness)), linestyles='solid',
                                       zorder=20, linewidths=4)
    ax.add_collection(line_segment_rows)
    fig = plt.gcf()

    # Color bar settings for Line Segments
    cbar = fig.colorbar(line_segment_rows,
                        ticks=[0,
                               norm(min(smoothness_of_segment1)),
                               norm(max(smoothness_of_segment1)),
                               norm(min(smoothness_of_segment2)),
                               norm(max(smoothness_of_segment2)),
                               1])  # Locations of labels on Color Bar
    cbar.set_label('Smoothness (%)', fontsize=20)
    cbar.ax.set_yticklabels(
        [str(color_bar_min),
         str(round(min(smoothness_of_segment1) * 100, 2)) + " - Min of " + name_segment_1,
         str(round(max(smoothness_of_segment1) * 100, 2)) + " - Max of " + name_segment_1,
         str(round(min(smoothness_of_segment2) * 100, 2)) + " - Min of " + name_segment_2,
         str(round(max(smoothness_of_segment2) * 100, 2)) + " - Max of " + name_segment_2,
         '100'], fontsize=16)  # Labels on Color Bar
    ax.autoscale()
########################################################################################################################

"""
# Use for personal plotting

import pandas as pd
import json

df = pd.read_csv('2D_Lattice.csv')
# row = 0
# box = df.iloc[row,1]
# array = np.array(json.loads(box))

# Select a subset of the data to use
number_samples = 10000
perplexity = 300

random_samples = sorted(np.random.randint(0,len(df), number_samples))  # Generates ordered samples

df = df.iloc[random_samples]

print(df)
print(np.shape(df))


# For plotting CSV data
# define a function to flatten a box
def flatten_box(box_str):
    box = json.loads(box_str)
    return np.array(box).flatten()


# apply the flatten_box function to each row of the dataframe and create a list of flattened arrays
flattened_arrays = df['Array'].apply(flatten_box).tolist()
avg_density = np.sum(flattened_arrays, axis=1)/(len(flattened_arrays[0]))

x, y, title, embedding = TSNE_reduction(flattened_arrays, perplexity=perplexity)
plot_dimensionality_reduction(x, y, avg_density, title)
plt.title(title)
plt.savefig('TSNE_Partial_Factorial_Perplexity_' + str(perplexity) + "_Data_Samples_" + str(number_samples))

"""