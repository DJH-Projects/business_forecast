# business_forecast

一个招投标项目的生命周期包含：预招标、招标、更改、澄清、预中标、中标、废标、流标等公告类型，而业务方对**同一个招投标项目**的整个生命周期都比较关心所以需要**同一个招投标项目**的更改、澄清、预中标、中标、废标、流标等公告类型的公告都关联到这个招投标项目的**招标公告**。

├── model 模型

│   	├── ./analyse_threshold.py 分析阈值
│   	├── ./annotate.py 标注数据
│   	├── ./main.py 预测商机
│   	├── ./score.py 在标注数据上测试模型指标

│   ├── test_pattern 测试模式
│   └── universal-sentence-encoder-multilingual_2 USE编码器
├── raw_data 数据
│   ├── history 历史文件
│   └── input 输入文件
└── utils 预处理工具
    ├── csv
    └── xls



