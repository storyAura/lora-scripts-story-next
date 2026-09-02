Schema.intersect([
    Schema.object({
        model_train_type: Schema.string().default("krea2-lora").hidden().description("训练种类"),
        lora_type: Schema.union(["lora", "lokr"]).default("lora").description("适配器类型"),
        pretrained_model_name_or_path: Schema.string().role('filepicker', { type: "model-file" }).default("./sd-models/krea2/raw.safetensors").description("Krea2 RAW DiT 路径（只在 RAW 上训 LoRA）"),
        vae: Schema.string().role('filepicker', { type: "model-file" }).default("./sd-models/krea2/qwen_image_vae.safetensors").description("Qwen-Image VAE 路径"),
        text_encoder: Schema.string().role('filepicker', { type: "model-file" }).default("./sd-models/krea2/qwen3_vl_4b_instruct.safetensors").description("Qwen3-VL-4B-Instruct 文本编码器"),
        resume: Schema.string().role('filepicker', { type: "folder" }).description("从某个 `save_state` 保存的中断状态继续训练，填写文件路径"),
    }).description("训练用模型"),

    Schema.object({
        timestep_sampling: Schema.union(["shift", "krea2_shift", "flux_shift", "sigmoid", "uniform"]).default("shift").description("时间步采样。shift=固定位移；krea2_shift=分辨率感知 μ（256px→0.5，1280px→1.15）；flux_shift 在 1024 提前饱和"),
        discrete_flow_shift: Schema.number().step(0.001).default(2.5).description("固定 Euler 流位移。仅 timestep_sampling=shift 时使用；选 krea2_shift 时请保持 2.5 或留空"),
        sigmoid_scale: Schema.number().step(0.001).default(1.0).description("sigmoid / shift 采样的缩放"),
        model_prediction_type: Schema.union(["raw", "additive", "sigma_scaled"]).default("raw").description("模型预测类型。Krea2 默认 raw"),
        weighting_scheme: Schema.union(["none", "uniform", "sigma_sqrt", "logit_normal", "mode", "cosmap"]).default("none").description("时间步损失加权。官方默认 none"),
        turbo_dit: Schema.boolean().default(false).description("预览按 Turbo 处理（钉死 μ=1.15，约 8 步，CFG 1）。训练仍应使用 RAW 底模"),
        sample_mu: Schema.number().step(0.01).min(0).description("预览 μ 覆盖。留空：RAW 按分辨率算，Turbo 用 1.15"),
    }).description("Krea2 专用参数"),

    Schema.object(
        UpdateSchema(SHARED_SCHEMAS.RAW.DATASET_SETTINGS, {
            resolution: Schema.string().default("1024,1024").description("训练图片分辨率，宽x高。支持非正方形，但必须是 64 倍数。"),
            enable_bucket: Schema.boolean().default(true).description("启用 arb 桶以允许非固定宽高比的图片"),
            min_bucket_reso: Schema.number().default(256).description("arb 桶最小分辨率"),
            max_bucket_reso: Schema.number().default(1280).description("arb 桶最大分辨率"),
            bucket_reso_steps: Schema.number().default(64).description("arb 桶分辨率划分单位，Krea2 需大于 32"),
        })
    ).description("数据集设置"),

    SHARED_SCHEMAS.SAVE_SETTINGS,

    Schema.object({
        max_train_epochs: Schema.number().min(1).default(20).description("最大训练 epoch（轮数）"),
        train_batch_size: Schema.number().min(1).default(1).description("批量大小, 越高显存占用越高"),
        gradient_checkpointing: Schema.boolean().default(true).description("梯度检查点"),
        gradient_accumulation_steps: Schema.number().min(1).default(1).description("梯度累加步数"),
        network_train_unet_only: Schema.boolean().default(true).description("仅训练 DiT"),
        blocks_to_swap: Schema.number().min(0).max(26).step(1).description("块交换数量，最大 26。与 turbo_dit 互斥"),
    }).description("训练相关参数"),

    SHARED_SCHEMAS.LR_OPTIMIZER,

    Schema.intersect([
        Schema.object({
            network_module: Schema.string().default("networks.lora_krea2").hidden(),
            lycoris_algo: Schema.string().hidden(),
            network_weights: Schema.string().role('filepicker').description("从已有的 LoRA 模型上继续训练，填写路径"),
            network_dim: Schema.number().min(1).default(32).description("网络维度。LoRA 常用 32；LoKr 里这是分解阈值，容量靠 factor"),
            network_alpha: Schema.number().min(1).default(32).description("常用值：LoRA 等于 network_dim；LoKr 全矩阵时会被忽略"),
            network_dropout: Schema.number().step(0.01).default(0).description("dropout 概率（LyCORIS 请用其自带 dropout）"),
            scale_weight_norms: Schema.number().step(0.01).min(0).description("最大范数正则化。如果使用，推荐为 1。LoKr 全矩阵留空时自动填 1"),
            network_args_custom: Schema.array(String).role('table').description("自定义 network_args，一行一个"),
        }).description("网络设置"),
        Schema.union([
            Schema.object({
                lora_type: Schema.const("lokr").required(),
                lokr_factor: Schema.number().min(-1).default(-1).description("LoKr 分解因子。-1 为最均衡（参数最少）"),
                full_matrix: Schema.boolean().default(false).description("全矩阵模式。留空 scale_weight_norms 时自动填 1.0"),
                lycoris_kernel_backend: Schema.union(["auto", "torch", "triton"]).default("auto").description("只影响 LyCORIS 算法的步进加速。推荐 auto：能用 Triton 就用，不行自动回退。要强制 Triton 选 triton；数值异常再改回 torch。bokr / bora / gsokr / glora_boft 不走这里"),
            }),
            Schema.object({}),
        ]),
    ]),

    Schema.intersect([
        Schema.object({
            enable_preview: Schema.boolean().default(false).description("启用训练预览图"),
        }).description("训练预览图设置"),
        Schema.union([
            Schema.object({
                enable_preview: Schema.const(true).required(),
                randomly_choice_prompt: Schema.boolean().default(false).description("随机选择预览图 Prompt"),
                prompt_file: Schema.string().role('textarea').description("预览图 Prompt 文件路径。填写后将采用文件内的 prompt，而下方的选项将失效。"),
                positive_prompts: Schema.string().role('textarea').default("masterpiece, best quality, 1girl, solo").description("Prompt。每行一条 = 一张预览图"),
                negative_prompts: Schema.string().role('textarea').default("lowres, bad anatomy, bad hands, text, error").description("Negative Prompt"),
                sample_width: Schema.number().default(1024).description("预览图宽"),
                sample_height: Schema.number().default(1024).description("预览图高"),
                sample_cfg: Schema.number().min(1).max(30).default(5.5).description("CFG Scale。RAW 建议 5.5，Turbo 为 1"),
                sample_seed: Schema.number().default(2333).description("种子"),
                sample_steps: Schema.number().min(1).max(300).default(28).description("迭代步数。RAW 约 28，Turbo 约 8"),
                sample_sampler: Schema.union(["euler"]).default("euler").description("Krea2 预览仅 euler"),
                sample_every_n_epochs: Schema.number().default(2).description("每 N 个 epoch 生成一次预览图"),
            }),
            Schema.object({}),
        ]),
    ]),

    SHARED_SCHEMAS.LOG_SETTINGS,

    Schema.object(UpdateSchema(SHARED_SCHEMAS.RAW.CAPTION_SETTINGS, {}, ["max_token_length"])).description("caption（Tag）选项"),

    SHARED_SCHEMAS.NOISE_SETTINGS,

    SHARED_SCHEMAS.DATA_ENCHANCEMENT,

    SHARED_SCHEMAS.OTHER,

    Schema.object(
        UpdateSchema(SHARED_SCHEMAS.RAW.PRECISION_CACHE_BATCH, {
            fp8_base: Schema.boolean().default(true).description("对基础模型使用 FP8 精度"),
            fp8_scaled: Schema.boolean().default(false).description("缩放 FP8。必须同时打开 fp8_base"),
            sdpa: Schema.boolean().default(true).description("启用 sdpa"),
            cache_text_encoder_outputs: Schema.boolean().default(true).description("缓存文本编码器的输出，减少显存使用。使用时需要关闭 shuffle_caption"),
            cache_text_encoder_outputs_to_disk: Schema.boolean().default(true).description("缓存文本编码器的输出到磁盘"),
        }, ["xformers"])
    ).description("速度优化选项"),

    SHARED_SCHEMAS.DISTRIBUTED_TRAINING
]);
