import os
import torch


def save_checkpoint(
    model,
    optimizer,
    epoch,
    checkpoint_path,
    criterion=None,
    scheduler=None,
    best_val_acc=None,
):
    """
    Lưu checkpoint an toàn hơn:
    - Ghi ra file .tmp trước
    - Ghi xong mới replace sang file .pth
    - Nếu lỗi giữa chừng thì không làm hỏng checkpoint cũ

    Lưu thêm:
    - criterion: ArcFaceLoss có tham số riêng
    - scheduler: để resume nếu sau này cần
    - best_val_acc: để biết checkpoint tốt nhất hiện tại
    """

    checkpoint_dir = os.path.dirname(checkpoint_path)

    if checkpoint_dir != "":
        os.makedirs(checkpoint_dir, exist_ok=True)

    temp_path = checkpoint_path + ".tmp"

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if criterion is not None:
        checkpoint["criterion_state_dict"] = criterion.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if best_val_acc is not None:
        checkpoint["best_val_acc"] = best_val_acc

    try:
        torch.save(checkpoint, temp_path)

        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        os.replace(temp_path, checkpoint_path)

    except Exception as error:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

        raise RuntimeError(
            f"Lỗi khi lưu checkpoint tại: {checkpoint_path}\n"
            f"Khả năng cao do ổ đĩa gần đầy, file bị khóa, hoặc không có quyền ghi.\n"
            f"Chi tiết lỗi: {error}"
        )


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

        if self.count > 0:
            self.avg = self.sum / self.count
        else:
            self.avg = 0.0