<h1>Hàm Mất Mát và Hàm Kích Hoạt trong PyTorch</h1>

<p>Đoạn mã dưới đây trình bày cách tính các hàm mất mát (loss functions) và hàm kích hoạt (activation functions) trong PyTorch. Chúng bao gồm CrossEntropy Loss, Mean Square Error, Binary Entropy Loss, cùng với các hàm kích hoạt như Sigmoid, ReLU, Softmax và Tanh.</p>

<h2>1. Tính Toán Loss Functions</h2>

<pre><code>import torch
import torch.nn.functional as F

# Công thức tính CrossEntropy Loss
def crossEntropyLoss(output, target):
    return F.cross_entropy(output, target)

# Công thức tính Mean Square Error
def meanSquareError(output, target):
    return torch.mean((output - target) ** 2)

# Công thức tính BinaryEntropy Loss
def binaryEntropyLoss(output, target, n):
    loss = -(1/n) * torch.sum(target * torch.log(output) + (1 - target) * torch.log(1 - output))
    return loss
</code></pre>

<p>Ở trên, chúng ta đã định nghĩa ba hàm mất mát:</p>
<ul>
    <li><strong>crossEntropyLoss:</strong> Tính toán Cross Entropy Loss, thường được sử dụng trong các bài toán phân loại đa lớp.</li>
    <li><strong>meanSquareError:</strong> Tính toán Mean Square Error, dùng để đo độ lệch giữa giá trị dự đoán và giá trị thực tế.</li>
    <li><strong>binaryEntropyLoss:</strong> Tính toán Binary Entropy Loss, được sử dụng cho các bài toán phân loại nhị phân.</li>
</ul>

<h2>2. Tính Toán Loss</h2>

<pre><code>inputs = torch.tensor([0.1, 0.3, 0.6, 0.7])
target = torch.tensor([0.31, 0.32, 0.8, 0.2])
n = len(inputs)

mse = meanSquareError(inputs, target)
binary_loss = binaryEntropyLoss(inputs, target, n)
cross_loss = crossEntropyLoss(inputs, target)

# In kết quả của các hàm Loss
print(f"Mean Square Error: {mse}")
print(f"Binary Entropy Loss: {binary_loss}")
print(f"Cross Entropy Loss: {cross_loss}")
</code></pre>

<p>Trong đoạn mã này, chúng ta sử dụng các hàm đã định nghĩa để tính toán giá trị của các hàm mất mát cho các đầu vào và mục tiêu cụ thể.</p>

<h2>3. Tính Toán Hàm Kích Hoạt</h2>

<pre><code># Công thức hàm sigmoid
def sigmoid(x: torch.tensor):
    return 1 / (1 + torch.exp(-x))

# Công thức hàm relu
def relu(x: torch.tensor):
    return torch.max(torch.tensor(0.0), x)

# Công thức hàm softmax
def softmax(zi: torch.tensor):
    exp_zi = torch.exp(zi)
    return exp_zi / torch.sum(exp_zi)

# Công thức hàm tanh
def tanh(x: torch.tensor):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
</code></pre>

<p>Ở trên, chúng ta định nghĩa các hàm kích hoạt:</p>
<ul>
    <li><strong>sigmoid:</strong> Chuyển đổi đầu vào về khoảng (0, 1).</li>
    <li><strong>relu:</strong> Trả về giá trị đầu vào nếu nó dương, ngược lại trả về 0.</li>
    <li><strong>softmax:</strong> Chuyển đổi đầu vào thành xác suất cho các lớp khác nhau.</li>
    <li><strong>tanh:</strong> Chuyển đổi đầu vào về khoảng (-1, 1).</li>
</ul>

<h2>4. Tính Toán Các Hàm Kích Hoạt</h2>

<pre><code># Tính toán các hàm activation với một tensor mẫu
x = torch.tensor([1.0, 5.0, -4.0, 3.0, -2.0])

f_sigmoid = sigmoid(x)
f_relu = relu(x)
f_softmax = softmax(x)
f_tanh = tanh(x)

# In kết quả của các hàm activation
print(f"Sigmoid = {f_sigmoid}")
print(f"Relu = {f_relu}")
print(f"Softmax = {f_softmax}")
print(f"Tanh = {f_tanh}")
</code></pre>

<p>Cuối cùng, chúng ta áp dụng các hàm kích hoạt lên một tensor mẫu và in ra kết quả.</p>
<h1>Mạng Neural Đơn Giản cho Dữ Liệu MNIST</h1>

<p>Đoạn mã này sử dụng PyTorch để xây dựng và huấn luyện một mạng neural đơn giản cho bài toán phân loại chữ số trong bộ dữ liệu MNIST. Mạng sẽ nhận đầu vào là các hình ảnh 28x28 pixel và dự đoán chữ số từ 0 đến 9.</p>

<h2>1. Thư Viện và Thiết Bị</h2>

<pre><code>import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import SGD
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms

# Use GPU if available, else use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
</code></pre>

<p>Đầu tiên, chúng ta nhập các thư viện cần thiết và xác định thiết bị (GPU hoặc CPU) để thực hiện tính toán.</p>

<h2>2. Tiền Xử Lý Dữ Liệu</h2>

<pre><code># Data transformation: Convert images to tensor and normalize them
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load the MNIST dataset
trainset = torchvision.datasets.MNIST(root='C:\\user2\\datasets\\MNIST', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=True)

testset = torchvision.datasets.MNIST(root='C:\\user2\\datasets\\MNIST', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False)
</code></pre>

<p>Chúng ta định nghĩa quy trình chuyển đổi dữ liệu để chuyển đổi hình ảnh thành tensor và chuẩn hóa chúng. Sau đó, bộ dữ liệu MNIST được tải xuống và chia thành tập huấn luyện và tập kiểm tra.</p>

<h2>3. Hiển Thị Hình Ảnh</h2>

<pre><code># Function to display images
def imshow(img):
    img = img * 0.5 + 0.5  # Denormalize the image
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()

# Display the first 8 images in the batch
for i, (images, labels) in enumerate(trainloader, 0):
    imshow(torchvision.utils.make_grid(images[:8]))
    break
</code></pre>

<p>Hàm <strong>imshow</strong> được sử dụng để hiển thị các hình ảnh từ bộ dữ liệu. Chúng ta hiển thị 8 hình ảnh đầu tiên trong batch huấn luyện.</p>

<h2>4. Tạo Mô Hình</h2>

<pre><code># Define the model
def getModel(n_features):
    model = nn.Sequential(
        nn.Flatten(),            # Flatten the 28x28 image into a vector of 784 elements
        nn.Linear(n_features, 256),  # First hidden layer with 256 neurons
        nn.ReLU(),               # Activation function
        nn.Linear(256, 10)       # Output layer (10 classes, one for each digit)
    ).to(device)
    return model

# Create the model
n_features = 28 * 28  # MNIST images are 28x28 pixels
model = getModel(n_features)
</code></pre>

<p>Hàm <strong>getModel</strong> định nghĩa một mô hình mạng neural với một lớp ẩn có 256 nơ-ron và một lớp đầu ra với 10 nơ-ron (mỗi nơ-ron tương ứng với một chữ số từ 0 đến 9).</p>

<h2>5. Định Nghĩa Tối Ưu và Hàm Mất Mát</h2>

<pre><code># Define optimizer and loss function
lr = 0.01
optim = SGD(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()
</code></pre>

<p>Chúng ta sử dụng phương pháp tối ưu SGD và hàm mất mát Cross Entropy cho bài toán phân loại đa lớp.</p>

<h2>6. Đánh Giá Mô Hình</h2>

<pre><code># Function to evaluate the model on the test set
def evaluate(model, testloader, criterion):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation during evaluation
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_loss = test_loss / len(testloader)
    return test_loss, accuracy
</code></pre>

<p>Hàm <strong>evaluate</strong> được định nghĩa để đánh giá mô hình trên tập kiểm tra, trả về giá trị mất mát và độ chính xác.</p>

<h2>7. Vòng Lặp Huấn Luyện</h2>

<pre><code>n_epochs = 10
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# Training loop
for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(trainloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)

        optim.zero_grad()  # Reset gradients

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        running_correct += (predicted == labels).sum().item()

        loss.backward()  # Backpropagation
        optim.step()     # Update weights

    # Calculate accuracy and loss for the epoch
    epoch_accuracy = 100 * running_correct / total
    epoch_loss = running_loss / (i + 1)
    test_loss, test_accuracy = evaluate(model, testloader, loss_fn)

    print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
</code></pre>

<p>Vòng lặp huấn luyện được thực hiện cho số lượng epochs xác định. Trong mỗi epoch, chúng ta tính toán mất mát, độ chính xác và cập nhật trọng số của mô hình.</p>

<h2>8. Đồ Thị Kết Quả Huấn Luyện</h2>

<pre><code>plt.figure(figsize=(20, 5))
plt.subplot(121)
plt.title('Loss Epochs')
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.subplot(122)
plt.title('Accuracy Epoch')
plt.plot(test_accuracies, label='Test Accuracy')
plt.legend()
plt.show()
</code></pre>

<p>Cuối cùng, chúng ta vẽ đồ thị hiển thị sự thay đổi của mất mát và độ chính xác trong quá trình huấn luyện.</p>

<h2>9. Lưu Mô Hình</h2>

<pre><code># Save the model weights
torch.save(model, "MLP_mnist.pth")
</code></pre>

<p>Mô hình được lưu lại để sử dụng cho các tác vụ sau này.</p>
