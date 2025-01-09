from multimodal_retriever_base import MultimodalRetrieverConfig, MultimodalRetriever
# 初始化配置
config = MultimodalRetrieverConfig(
    model_name='ViT-B-16',
    index_path='./index',
    batch_size=32,
    dim=512,
    download_root="data/chinese-clip-vit-base-patch16/"
)

# 创建检索器示例
retriever = MultimodalRetriever(config)


## %%time
ex = [{"image_base64": "", "caption": ""}]
# 对 examples构建索引
img_text_pairs = [(ex['image_base64'], ex['caption']) for ex in examples]
retriever.build_from_pairs(img_text_pairs)

retriever.save_index()
