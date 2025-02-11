import cv2
import numpy as np
import matplotlib.pyplot as plt


def downsample_image(img, scale_factor):
    height, width = img.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image


def upsample_image(img, scale_factor):
    height, width = img.shape[:2]
    new_width = int(width / scale_factor)
    new_height = int(height / scale_factor)
    resized_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image


def cartoonize(img, k):
    # Defining input data for clustering
    data = np.float32(img).reshape((-1, 3))
    # Defining criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # Applying cv2.kmeans function
    _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    print(center)
    # Reshape the output data to the size of input image
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    # Smooth the result
    return result


# Reading image
img = cv2.imread("Sliki/imageProba.png")
height, width = img.shape[:2]
max_dim = max(height, width)
scaling_factor = 1
if max_dim > 1000:
    scaling_factor = 1000 / max_dim
    img = downsample_image(img, scaling_factor)


#2.Odreduvanje na rabovi
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Convert the input image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# Perform adaptive threshold
edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8)

# Show the output
plt.imshow(img, cmap='gray')
plt.axis("off")
plt.title("START")
plt.show()

result = cartoonize(img, 12)
# Smooth the result
blurred = cv2.medianBlur(result, 5)

# Combine the result and edges to get final cartoon effect
cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)

# Upsample the cartoonized image to the original size
cartoon_upsampled = upsample_image(cartoon, scaling_factor)

plt.imshow(cartoon, cmap='gray')
plt.axis("off")
plt.title("FINAL")
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()