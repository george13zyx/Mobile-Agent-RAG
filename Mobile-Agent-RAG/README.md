## Mobile-Agent-RAG: Driving Smart Multi-Agent Coordination with Contextual Knowledge Empowerment for Long-Horizon Mobile Automation

### Preparation

#### 1. Android Debug Bridge 

1. Download the [Android Debug Bridge](https://developer.android.com/tools/releases/platform-tools?hl=en).
2. Turn on the ADB debugging switch on your Android phone, it needs to be turned on in the developer options first.
3. Connect your phone to the computer with a data cable and select "Transfer files".
4. Test your ADB environment as follow: `/path/to/adb devices`. If the connected devices are displayed, the preparation is complete.
5. If you are using a MAC or Linux system, make sure to turn on adb permissions as follow: `sudo chmod +x /path/to/adb`
6. If you are using Windows system, the path will be `xx/xx/adb.exe`

#### 2. Install the ADB Keyboard on your Mobile Device

1. Download the ADB keyboard [apk](https://github.com/senzhk/adbkeyboard/blob/master/adbkeyboard.apk) installation package.
2. Click the apk to install on your mobile device.
3. Switch the default input method in the system settings to "ADB Keyboard".

#### 3. Openai/Gemini/Claude API keys setting

```bash
export ADB_PATH="your/path/to/adb"
export BACKBONE_TYPE="OpenAI"
export OPENAI_API_KEY="your-openai-key"
```

```bash
export ADB_PATH="your/path/to/adb"
export BACKBONE_TYPE="Gemini"
export GEMINI_API_KEY="your-gemini-key"
```

```bash
export ADB_PATH="your/path/to/adb"
export BACKBONE_TYPE="Claude"
export CLAUDE_API_KEY="your-claude-key"
```



### Running

#### 1. Agent Installation

```bash
conda create -n agent python=3.10.16
conda activate agent
pip install -r requirements_agent.txt
```

#### 2. RAG Installation

```bash
conda create -n rag python=3.8.18
conda activate rag
pip install -r requirements_rag.txt
```

#### 3. RAG Setup

Terminal 1:

```bash
conda activate rag
python passage_retrieval_manager_server.py
```

Terminal 2:

```
conda activate rag
python passage_retrieval_operator_server.py
```

#### **4.RAG Knowledge Base Construction**

Manager-RAG Knowledge Base Construction:

```bash
conda activate rag
bash /data0/zhouyuxiang/mobile_eval_e_retrieve/manager/generate_embedding.sh
```

Operator-RAG Knowledge Base Construction:

```bash
conda activate rag
bash mobile_eval_e_retrieve/operator/app/generate_embedding.sh
```

#### 5.Run Mobile-Agent-RAG

```bash
conda activate agent
bash run_task.sh
```

