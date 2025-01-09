from multimodal_retriver_base import MultimodalRetrieverConfig, MultimodalRetriever
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

# 加载索引
retriever.load_index()


# 使用文本检索
query_text = "飞机"
results = retriever.retrieve(query_text, top_k=3)
results


# 测试图片检索
examples = [{"image_base64": ""}]
query_image = retriever.convert_base642image(examples[0]['image_base64'])
results = retriever.retrieve(query_image, top_k=3)
results
