import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from model import FaceEmbeddingModel
from transforms import eval_transform


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--query_image", type=str, required=True)
    parser.add_argument("--gallery_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["resnet50", "mobilenetv2"],
        required=True
    )

    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--pretrained", action="store_true")

    return parser.parse_args()


def collect_gallery_images(gallery_dir: Path):
    image_paths = []

    for path in sorted(gallery_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_paths.append(path)

    return image_paths


def load_model(args, device):
    model = FaceEmbeddingModel(
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        pretrained=args.pretrained
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint_path,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


@torch.no_grad()
def extract_embedding(model, image_path: Path, device):
    image = Image.open(image_path).convert("RGB")
    tensor = eval_transform(image).unsqueeze(0).to(device)

    embedding = model(tensor)
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

    return embedding[0].cpu().numpy()


def make_montage(query_path, results, output_path, thumb_size=160):
    cols = 5

    result_rows = int(np.ceil(len(results) / cols))
    rows = 1 + result_rows

    title_h = 35
    text_h = 45
    cell_w = thumb_size
    cell_h = thumb_size + text_h

    canvas_w = cols * cell_w
    canvas_h = title_h + rows * cell_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((10, 10), "Query image", fill=(0, 0, 0))

    query_img = Image.open(query_path).convert("RGB").resize((thumb_size, thumb_size))
    canvas.paste(query_img, (0, title_h))
    draw.text((5, title_h + thumb_size + 5), "QUERY", fill=(0, 0, 0))
    draw.text((5, title_h + thumb_size + 23), Path(query_path).parent.name, fill=(80, 80, 80))

    draw.text((thumb_size + 10, 10), f"Top-{len(results)} Retrieval Results", fill=(0, 0, 0))

    start_y = title_h + cell_h

    for idx, item in enumerate(results):
        row = idx // cols
        col = idx % cols

        x = col * cell_w
        y = start_y + row * cell_h

        img = Image.open(item["image_path"]).convert("RGB").resize((thumb_size, thumb_size))
        canvas.paste(img, (x, y))

        person_name = Path(item["image_path"]).parent.name

        draw.text((x + 5, y + thumb_size + 5), f"#{item['rank']} sim={item['score']:.3f}", fill=(0, 0, 0))
        draw.text((x + 5, y + thumb_size + 23), person_name, fill=(80, 80, 80))

    canvas.save(output_path)


def main():
    args = parse_args()

    query_image = Path(args.query_image)
    gallery_dir = Path(args.gallery_dir)
    checkpoint_path = Path(args.checkpoint_path)
    output_path = Path(args.output_path)

    if not query_image.exists():
        raise FileNotFoundError(f"Không tìm thấy query_image: {query_image}")

    if not gallery_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy gallery_dir: {gallery_dir}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("DEMO RETRIEVAL")
    print("=" * 80)
    print("Query image:", query_image)
    print("Gallery dir:", gallery_dir)
    print("Checkpoint :", checkpoint_path)
    print("Model      :", args.model_name)
    print("Device     :", device)
    print("=" * 80)

    model = load_model(args, device)

    gallery_images = collect_gallery_images(gallery_dir)

    if len(gallery_images) == 0:
        raise RuntimeError("Gallery không có ảnh.")

    query_embedding = extract_embedding(
        model=model,
        image_path=query_image,
        device=device
    )

    gallery_embeddings = []

    print(f"Extracting gallery embeddings: {len(gallery_images)} images")

    for image_path in gallery_images:
        emb = extract_embedding(
            model=model,
            image_path=image_path,
            device=device
        )
        gallery_embeddings.append(emb)

    gallery_embeddings = np.vstack(gallery_embeddings)

    similarities = gallery_embeddings @ query_embedding

    # Nếu query nằm trong gallery, loại chính nó ra khỏi kết quả
    for idx, image_path in enumerate(gallery_images):
        if image_path.resolve() == query_image.resolve():
            similarities[idx] = -999.0

    top_indices = np.argsort(-similarities)[:args.top_k]

    results = []

    print()
    print("TOP-K RESULTS")
    print("=" * 80)

    for rank, idx in enumerate(top_indices, start=1):
        image_path = gallery_images[idx]
        score = float(similarities[idx])

        print(f"#{rank:<2} score={score:.4f} | {image_path}")

        results.append({
            "rank": rank,
            "image_path": str(image_path),
            "score": score,
        })

    make_montage(
        query_path=query_image,
        results=results,
        output_path=output_path
    )

    print("=" * 80)
    print(f"Đã lưu ảnh demo: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()