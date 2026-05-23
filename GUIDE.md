1. Cài thư viện
    pip install torch torchvision tqdm tensorboard pillow
    NOTE: chạy mà thiếu gì thì cài thêm nhé=))

2. Các file chính
   File	                         Chức năng
dataset.py	                   Load dataset
transforms.py	                Augmentation
model.py	              ResNet50 và MobileNetv2 (chọn model để train trong file train.py)
losses.py	                   ArcFace Loss
train.py	                    Train model
utils.py	            Save checkpoint + AverageMeter

3. Các args hỗ trợ
    Arg	                           Ý nghĩa
--data_dir	                   Folder dataset
--batch_size	                 Batch size
--epochs	                      Số epoch
--lr	                        Learning rate
--embedding_dim	             Kích thước embedding
--num_workers	              DataLoader workers
--checkpoint_dir	           Folder lưu model
--log_dir	                 Folder tensorboard

LƯU Ý: các args này có giá trị mặc định. Nếu muốn thay đổi thì cứ thay đổi thoải mái.

4. TensorBoard
Chạy: tensorboard --logdir=logs (chạy trong terminal của pycharm luôn cho nhanh)
Mở trình duyệt: http://localhost:6006

Xem: loss, learning rate, training progress

5. Checkpoint
Model sẽ được lưu tại:
    checkpoints/
    ├── epoch_1.pth
    ├── epoch_2.pth
    └── ...

6. Output model
Model học sinh ra:
    face image
        ↓
    embedding vector (512-dim)
Embedding này dùng để: face search, face clustering, tìm ảnh cùng người, event retrieval, ..v..v...(như
của ta là tìm ảnh cùng người)

7. Pipeline hoạt động
    Face Image
        ↓
    ResNet50
        ↓
    512-D Embedding
        ↓
    ArcFace Loss
        ↓
    Embedding cùng người gần nhau
    Embedding khác người xa nhau

8. Mục tiêu cuối cùng
Sau khi train xong:
    Input:
    1 ảnh mặt người
    
    Output:
    Toàn bộ ảnh của người đó
    trong mọi event

9. Sau khi có baseline tốt
mới thêm:
EarlyStopping
TTA
FAISS
hard negative mining
re-ranking.