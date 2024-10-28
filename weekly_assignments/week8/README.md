<h1>Giải Thích Chi Tiết Mã Nguồn</h1>

<h2>1. Thư Viện và Thiết Bị</h2>
<pre><code>import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Adam
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
</code></pre>

<ul>
    <li><strong>Thư Viện PyTorch</strong>: <code>torch</code> là thư viện chính để xây dựng mô hình học sâu. Nó cung cấp các lớp để tạo mạng nơ-ron, xử lý tensor (ma trận n chiều), và thực hiện tính toán trên GPU.</li>
    <li><strong>Thư Viện Đồ Thị</strong>: <code>matplotlib.pyplot</code> (<code>plt</code>) cho phép bạn vẽ đồ thị và hình ảnh, rất hữu ích để trực quan hóa kết quả.</li>
    <li><strong>Thư Viện NumPy</strong>: <code>numpy</code> là thư viện để xử lý mảng và toán học.</li>
    <li><strong>Thư Viện Tối Ưu</strong>: <code>torch.optim</code> cung cấp các thuật toán tối ưu hóa như Adam, giúp cải thiện độ chính xác của mô hình.</li>
    <li><strong>Thư Viện Mạng Nơ-Ron</strong>: <code>torch.nn</code> là thư viện giúp định nghĩa cấu trúc mạng nơ-ron.</li>
    <li><strong>Thư Viện Datasets và Transforms</strong>: <code>torchvision.datasets</code> và <code>torchvision.transforms</code> giúp tải dữ liệu và thực hiện các biến đổi hình ảnh (như chuẩn hóa, tăng cường dữ liệu).</li>
    <li><strong>DataLoader</strong>: Đây là lớp cho phép bạn tải dữ liệu theo lô (batch), giúp xử lý dễ dàng hơn trong quá trình huấn luyện.</li>
</ul>

<h2>2. Xác Định Thiết Bị</h2>
<pre><code>device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
</code></pre>

<ul>
    <li><strong>Xác Định Thiết Bị</strong>: Dòng mã này kiểm tra xem GPU (CUDA) có sẵn không. Nếu có, nó sẽ sử dụng GPU để tính toán, giúp tăng tốc quá trình huấn luyện. Nếu không, nó sẽ sử dụng CPU.</li>
</ul>

<h2>3. Tải Dữ Liệu CIFAR10</h2>
<pre><code>transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
</code></pre>

<ul>
    <li><strong>Biến Đổi Dữ Liệu</strong>: <code>transforms.Compose</code> là danh sách các biến đổi sẽ được áp dụng lên hình ảnh.
        <ul>
            <li><code>RandomHorizontalFlip()</code>: Lật hình ảnh ngang một cách ngẫu nhiên. Điều này giúp mô hình trở nên mạnh mẽ hơn với các biến thể hình ảnh.</li>
            <li><code>RandomRotation(10)</code>: Xoay hình ảnh ngẫu nhiên trong khoảng từ -10 đến +10 độ.</li>
            <li><code>ToTensor()</code>: Chuyển đổi hình ảnh thành tensor PyTorch.</li>
            <li><code>Normalize()</code>: Chuẩn hóa dữ liệu về khoảng giá trị trung bình và độ lệch chuẩn cụ thể.</li>
        </ul>
    </li>
</ul>

<pre><code>trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
</code></pre>

<ul>
    <li><strong>Tải Dữ Liệu</strong>: Tải tập dữ liệu CIFAR-10 từ internet (nếu chưa có) và áp dụng các biến đổi đã định nghĩa.</li>
    <li><strong>DataLoader</strong>: Chia dữ liệu thành các lô (batch) với kích thước 64. <code>shuffle=True</code> giúp ngẫu nhiên hóa thứ tự dữ liệu, và <code>num_workers=2</code> cho phép tải dữ liệu song song, tăng tốc quá trình.</li>
</ul>

<h2>4. Mô Hình MLP</h2>
<pre><code>class MLPModel(nn.Module):
    def __init__(self, n_features):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(n_features, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
</code></pre>

<ul>
    <li><strong>Định Nghĩa Mô Hình</strong>: <code>MLPModel</code> là lớp mô hình mạng nơ-ron đa lớp (MLP). Nó kế thừa từ lớp <code>nn.Module</code>.</li>
    <li><strong>Khởi Tạo Mô Hình</strong>:
        <ul>
            <li><code>nn.Linear(n_features, 512)</code>: Lớp nơ-ron đầy đủ đầu vào từ số lượng đặc trưng (<code>n_features</code>) đến 512 nơ-ron.</li>
            <li><code>Dropout</code>: Giúp ngăn ngừa overfitting bằng cách bỏ qua 50% các nơ-ron trong mỗi lần huấn luyện.</li>
            <li><code>fc2</code> và <code>fc3</code>: Các lớp nơ-ron tiếp theo để giảm dần số lượng nơ-ron và cuối cùng đưa ra 10 đầu ra (cho 10 lớp trong CIFAR-10).</li>
        </ul>
    </li>
    <li><strong>Phương Thức <code>forward</code></strong>:
        <ul>
            <li><code>x.view(-1, 3 * 32 * 32)</code>: Chuyển đổi hình ảnh (3 kênh màu và kích thước 32x32) thành một vector. <code>-1</code> tự động xác định kích thước lô.</li>
            <li><code>torch.relu</code>: Hàm kích hoạt ReLU (Rectified Linear Unit), giúp tăng cường tính phi tuyến tính của mô hình.</li>
        </ul>
    </li>
</ul>

<h2>5. Khởi Tạo Loss Function và Optimizer</h2>
<pre><code>n_features = 3 * 32 * 32
model = MLPModel(n_features).to(device)
lr = 0.001
optim = Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()
</code></pre>

<ul>
    <li><strong>Số Lượng Đặc Trưng</strong>: Đối với hình ảnh CIFAR-10, mỗi hình ảnh có 3 kênh màu (RGB) và kích thước 32x32, do đó <code>n_features</code> là 3 * 32 * 32.</li>
    <li><strong>Khởi Tạo Mô Hình</strong>: Mô hình được khởi tạo và chuyển đến thiết bị (GPU hoặc CPU).</li>
    <li><strong>Learning Rate</strong>: <code>lr = 0.001</code> xác định tốc độ học. Tốc độ học quá cao có thể khiến mô hình không hội tụ, trong khi quá thấp có thể làm chậm quá trình.</li>
    <li><strong>Optimizer</strong>: Sử dụng Adam để tối ưu hóa các tham số của mô hình.</li>
    <li><strong>Loss Function</strong>: <code>CrossEntropyLoss</code> là hàm mất mát thường được sử dụng cho bài toán phân loại đa lớp. Nó đo lường độ khác biệt giữa nhãn thực và nhãn dự đoán của mô hình.</li>
</ul>

<h2>6. Hàm Đánh Giá</h2>
<pre><code>def evaluate(model, testloader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return test_loss / len(testloader), correct / total
</code></pre>

<ul>
    <li><strong>Đánh Giá Mô Hình</strong>: Hàm này thực hiện quá trình đánh giá mô hình trên tập kiểm tra.</li>
    <li><strong>Chế Độ Đánh Giá</strong>: <code>model.eval()</code> chuyển mô hình sang chế độ đánh giá, tắt dropout.</li>
    <li><strong>Không Tính Gradient</strong>: <code>with torch.no_grad()</code> giúp tiết kiệm bộ nhớ và tăng tốc độ tính toán trong quá trình đánh giá.</li>
    <li><strong>Tính Toán Mất Mát</strong>: Mất mát trên tập kiểm tra được tính tổng và chia cho số lượng lô để tính trung bình.</li>
    <li><strong>Tính Độ Chính Xác</strong>: So sánh dự đoán của mô hình với nhãn thực tế để tính độ chính xác.</li>
</ul>

<h2>7. Huấn Luyện Mô Hình</h2>
<pre><code>def train(model, trainloader, optimizer, criterion, device, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(trainloader):.4f}')
</code></pre>

<ul>
    <li><strong>Huấn Luyện Mô Hình</strong>: Hàm này thực hiện quá trình huấn luyện mô hình trên tập dữ liệu huấn luyện.</li>
    <li><strong>Chế Độ Huấn Luyện</strong>: <code>model.train()</code> chuyển mô hình sang chế độ huấn luyện.</li>
    <li><strong>Đặt Lại Gradient</strong>: <code>optimizer.zero_grad()</code> đặt lại gradient trước khi thực hiện tính toán mới.</li>
    <li><strong>Tính Mất Mát và Cập Nhật Tham Số</strong>: Tính toán mất mát và thực hiện bước tối ưu hóa bằng cách gọi <code>loss.backward()</code> và <code>optimizer.step()</code>.</li>
    <li><strong>In Kết Quả</strong>: In ra mất mát trung bình sau mỗi epoch.</li>
</ul>

<h2>8. Chạy Mô Hình</h2>
<pre><code>testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

train(model, trainloader, optim, loss_fn, device, epochs=10)
test_loss, test_accuracy = evaluate(model, testloader, loss_fn)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
</code></pre>

<ul>
    <li><strong>Tải Tập Kiểm Tra</strong>: Tương tự như tập huấn luyện, nhưng với <code>train=False</code> để chỉ tải tập kiểm tra.</li>
    <li><strong>Huấn Luyện Mô Hình</strong>: Gọi hàm <code>train</code> để huấn luyện mô hình trong 10 epoch.</li>
    <li><strong>Đánh Giá Mô Hình</strong>: Gọi hàm <code>evaluate</code> để tính toán mất mát và độ chính xác trên tập kiểm tra.</li>
    <li><strong>In Kết Quả Cuối Cùng</strong>: In ra mất mát và độ chính xác của mô hình trên tập kiểm tra.</li>
</ul>

</body>
</html>
