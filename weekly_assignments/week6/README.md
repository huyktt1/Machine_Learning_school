<h2>Support Vector Machine with Quadratic Programming</h2>

<p>Đoạn mã này sử dụng thư viện <code>cvxopt</code> để giải bài toán tối ưu hóa bậc hai (QP) nhằm xác định hyperplane phân tách trong một bài toán phân loại nhị phân.</p>

<h3>1. Nhập các thư viện cần thiết</h3>
<pre><code>from cvxopt import matrix as matrix
from cvxopt import solvers as solvers
import numpy as np
import matplotlib.pyplot as plt
</code></pre>
<p>Chúng ta nhập các thư viện <code>cvxopt</code> để tối ưu hóa, <code>numpy</code> để xử lý mảng, và <code>matplotlib</code> để trực quan hóa dữ liệu.</p>

<h3>2. Dữ liệu đầu vào</h3>
<pre><code># 3 data points
x = np.array([[1., 3.], [2., 2.], [1., 1.]])
y = np.array([[1.], [1.], [-1.]])
</code></pre>
<p>Chúng ta định nghĩa 3 điểm dữ liệu <code>x</code> và nhãn tương ứng <code>y</code>. Điểm có nhãn 1 được phân loại là lớp dương, trong khi điểm có nhãn -1 thuộc lớp âm.</p>

<h3>3. Tính toán ma trận H</h3>
<pre><code># ---- Calculate lambda using cvxopt ----
# Calculate H matrix
H = np.dot(y, y.T) * np.dot(x, x.T)  # Gram matrix with label influence
</code></pre>
<p>Chúng ta tính toán ma trận <code>H</code>, đây là ma trận Gram ảnh hưởng bởi nhãn.</p>

<h3>4. Thiết lập bài toán tối ưu hóa</h3>
<pre><code># Construct the matrices required for QP in standard form
n = x.shape[0]
P = matrix(H)
q = matrix(-np.ones((n, 1)))
G = matrix(-np.eye(n))  # λ >= 0
h = matrix(np.zeros(n))
A = matrix(y.reshape(1, -1))
b = matrix(np.zeros(1))
</code></pre>
<p>Chúng ta thiết lập các ma trận cần thiết để giải bài toán QP: <code>P</code>, <code>q</code>, <code>G</code>, <code>h</code>, <code>A</code>, và <code>b</code>.</p>

<h3>5. Giải bài toán QP</h3>
<pre><code># solver parameters
solvers.options['abstol'] = 1e-10
solvers.options['reltol'] = 1e-10
solvers.options['feastol'] = 1e-10

# Perform QP
sol = solvers.qp(P, q, G, h, A, b)
</code></pre>
<p>Chúng ta thiết lập các tham số cho trình giải và thực hiện giải bài toán tối ưu hóa QP để tìm <code>λ</code>.</p>

<h3>6. Tính toán vector trọng số (w) và độ lệch (b)</h3>
<pre><code># Calculate w using the lambda, which is the solution to QP
lamb = np.array(sol['x'])
w = np.sum(lamb * y * x, axis=0)

# Find support vectors
sv_idx = np.where(lamb > 1e-5)[0]  # λ > 0 => support vectors
sv_lamb = lamb[sv_idx]
sv_x = x[sv_idx]
sv_y = y[sv_idx].reshape(1, -1)

# Calculate b using the support vectors and calculate the average
b = sv_y[0, 0] - np.dot(w, sv_x[0])
b = np.mean([sv_y[0, i] - np.dot(w, sv_x[i]) for i in range(len(sv_x))])
</code></pre>
<p>Chúng ta tính toán vector trọng số <code>w</code> và độ lệch <code>b</code> bằng cách sử dụng các vector hỗ trợ.</p>

<h3>7. In kết quả và trực quan hóa</h3>
<pre><code># Visualize the data points
plt.figure(figsize=(5, 5))
color = ['red' if a == 1 else 'blue' for a in y]
plt.scatter(x[:, 0], x[:, 1], s=200, c=color, alpha=0.7)

# Visualize the decision boundary
x1_dec = np.linspace(0, 4, 100)
x2_dec = -(w[0] * x1_dec + b) / w[1]
plt.plot(x1_dec, x2_dec, c='black', lw=1.0, label='decision boundary')

# Visualize the positive & negative boundary
...
plt.legend()
plt.savefig('decision_boundary.png')
plt.close()  # Đóng biểu đồ sau khi lưu
</code></pre>
<p>Cuối cùng, chúng ta trực quan hóa các điểm dữ liệu, đường biên quyết định, và các biên tích cực/tiêu cực. Hình ảnh được lưu lại dưới tên <code>decision_boundary.png</code>.</p>

<h3>8. Lưu kết quả vào tệp</h3>
<pre><code># Lưu kết quả vào tệp results.txt
with open('results_1.txt', 'w') as f:
    f.write('Lambda:\n')
    f.write(np.array2string(np.round(lamb.flatten(), 3), separator=', '))
    f.write('\n\nWeight (w):\n')
    f.write(np.array2string(np.round(w, 3), separator=', '))
    f.write('\n\nBias (b):\n')
    f.write(np.array2string(np.round(b, 3), separator=', '))
</code></pre>
<p>Chúng ta lưu kết quả của <code>λ</code>, <code>w</code> và <code>b</code> vào tệp <code>results_1.txt</code>.</p>

<h2>LAB2 : Support Vector Machine (SVM) with Quadratic Programming</h2>
<p>Đoạn mã này sử dụng thư viện <code>cvxopt</code> để thực hiện tối ưu hóa bậc hai, nhằm tìm ra đường biên phân cách cho dữ liệu hai lớp.</p>

<pre><code>import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import matplotlib.pyplot as plt
</code></pre>

<h3>1. Dữ liệu huấn luyện</h3>
<pre><code># Dữ liệu huấn luyện
x = np.array([[0.2, 0.869], [0.687, 0.212], ...])  # Dữ liệu đầu vào
y = np.array([-1, 1, 1, 1, ...])  # Nhãn lớp tương ứng
y = y.astype('float').reshape(-1, 1)
</code></pre>
<p>Mảng <code>x</code> chứa các điểm dữ liệu với hai đặc trưng, và mảng <code>y</code> chứa nhãn lớp tương ứng cho mỗi điểm dữ liệu.</p>

<h3>2. Tính toán lambda bằng cvxopt</h3>
<pre><code># ---- Tính toán lambda sử dụng cvxopt ----
C = 50.0  # Hệ số điều chỉnh
N = x.shape[0]  # Số lượng điểm dữ liệu

# Xây dựng ma trận H
H = (y @ y.T) * (x @ x.T)
P = cvxopt_matrix(H)
q = cvxopt_matrix(np.ones(N) * -1)
A = cvxopt_matrix(y.reshape(1, -1))
b = cvxopt_matrix(np.zeros(1))

g = np.vstack([-np.eye(N), np.eye(N)])
G = cvxopt_matrix(g)
h1 = np.hstack([np.zeros(N), np.ones(N) * C])
h = cvxopt_matrix(h1)

# Tham số cho bộ giải
cvxopt_solvers.options['abstol'] = 1e-10
cvxopt_solvers.options['reltol'] = 1e-10
cvxopt_solvers.options['feastol'] = 1e-10

# Thực hiện QP
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
</code></pre>
<p>Đoạn mã trên xây dựng và giải bài toán tối ưu hóa bậc hai (QP) bằng cách sử dụng cvxopt. Kết quả là giá trị lambda cho từng điểm dữ liệu.</p>

<h3>3. Tính toán vector w và b</h3>
<pre><code># Tính toán w từ lambda
lamb = np.array(sol['x'])
w = np.sum(lamb * y * x, axis=0)

# Tìm các vector hỗ trợ
sv_idx = np.where(lamb > 1e-5)[0]
sv_lamb = lamb[sv_idx]
sv_x = x[sv_idx]
sv_y = y[sv_idx]

# Tính toán b từ các vector hỗ trợ
b = np.mean(sv_y - np.dot(sv_x, w))
</code></pre>
<p>Vector <code>w</code> được tính toán từ giá trị lambda, và <code>b</code> được tính từ các vector hỗ trợ, cho phép xác định đường biên phân cách.</p>

<h3>4. Hiển thị kết quả</h3>
<pre><code># Hiển thị đường biên phân cách
plt.figure(figsize=(7, 7))
color = ['red' if a == 1 else 'blue' for a in y]
plt.scatter(x[:, 0], x[:, 1], s=200, c=color, alpha=0.7)

# Hiển thị quyết định biên
x1_dec = np.linspace(0, 1, 100)
x2_dec = -(w[0] * x1_dec + b) / w[1]
plt.plot(x1_dec, x2_dec, c='black', lw=1.0, label='decision boundary')
</code></pre>
<p>Đoạn mã này sử dụng matplotlib để hiển thị dữ liệu huấn luyện và đường biên phân cách. Màu sắc của các điểm dữ liệu được phân loại theo nhãn lớp.</p>

<h3>5. Tính toán và hiển thị biên độ</h3>
<pre><code># Tính toán và hiển thị biên độ
w_norm = np.linalg.norm(w)
w_unit = w / w_norm
half_margin = 1 / w_norm
plt.title('C = ' + str(C) + ',  Σξ = ' + str(np.sum(slack).round(2)))
plt.show()
</code></pre>
<p>Cuối cùng, mã này hiển thị biên độ và tổng độ trễ <code>Σξ</code>.</p>

<h2>LAB3 : Dự đoán lớp cho các điểm mới</h2>
<p>Đoạn mã thứ hai mở rộng từ đoạn mã đầu tiên, thêm chức năng dự đoán lớp cho các điểm mới dựa trên vector w và b đã tính toán.</p>

<pre><code>def predict(x_new):
    return np.sign(np.dot(w, x_new) + b)
</code></pre>
<p>Hàm <code>predict</code> sử dụng vector <code>w</code> và <code>b</code> để xác định lớp cho các điểm mới.</p>

<h3>1. Dữ liệu mới và dự đoán</h3>
<pre><code># Dữ liệu mới để dự đoán
new_points = np.array([[0.4, 0.5], [0.7, 0.8], [0.1, 0.2]])
predictions = [predict(p) for p in new_points]
</code></pre>
<p>Đoạn mã trên kiểm tra dự đoán cho một số điểm mới và lưu trữ kết quả.</p>

<h3>2. Hiển thị dự đoán</h3>
<pre><code># Hiển thị kết quả dự đoán
plt.figure(figsize=(7, 7))
...
plt.title(f'Predictions for new points: {predictions}')
plt.show()
</code></pre>
<p>Cuối cùng, đoạn mã này hiển thị các điểm mới và màu sắc tương ứng với dự đoán của chúng.</p>

<h3>Kết quả</h3>
<pre><code>result3 = print("\nMargin = {:.4f}".format(half_margin * 2))
</code></pre>
<p>Kết quả cuối cùng là biên độ được tính toán và hiển thị.</p>
