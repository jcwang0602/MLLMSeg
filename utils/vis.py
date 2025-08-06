import matplotlib.pyplot as plt


def save_mask_for_debug(mask1, mask2, idx):
    # 创建一个图形对象
    plt.figure(figsize=(12, 8))
    # 第一张图: 显示第一个掩码，使用红色调色板
    plt.subplot(1, 2, 1)  # 1行2列的第一个位置
    plt.imshow(mask1.cpu().detach().numpy(), vmin=0, vmax=1)
    plt.title("Mask 1")
    plt.axis("off")  # 关闭坐标轴

    # 第二张图: 显示第二个掩码，使用绿色调色板
    plt.subplot(1, 2, 2)  # 1行2列的第二个位置
    plt.imshow(mask2.cpu().detach().numpy(), vmin=0, vmax=1)
    plt.title("Mask 2")
    plt.axis("off")  # 关闭坐标轴

    # 如果你还想保存图像
    plt.savefig(f"vis/{idx}_combined_masks.png", bbox_inches="tight", pad_inches=0)
