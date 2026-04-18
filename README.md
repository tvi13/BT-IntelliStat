## 📊 BT IntelliStat v1.0
Resilient Multi-Dataset Research & AI-Driven Statistical Analysis

BT IntelliStat is an analytical platform built for researchers to synthesize complex datasets through high-fidelity statistical modeling and multi-tier AI reasoning.

## 🚀 Version 1.0
A robust web service capable of handling real-world data constraints:

Triple-Tier AI Failover (The "Waterfall" Logic): Engineered a resilient inference system that prioritizes Gemini 2.5 Flash, automatically fails over to Gemini 2.5 Flash-Lite, and utilizes Groq (Llama 3.3 70B) as a tertiary high-speed backup.

High-Capacity Data Handling: Integrated a Sampling Guard and manual Garbage Collection routine to process datasets exceeding 10,000 rows, preventing RAM-related crashes on cloud infrastructure.

Persistent Research History: Configured a Persistent SSD Volume mapping to ensure analysis history and user databases survive application redeployments.

Optimized Server Performance: Deployed with a custom Gunicorn configuration using threaded workers timeout to support intensive comparative computations.

## ✨ Core Features
Comparative Engine: Compare up to 3 datasets with automatic source-file lineage tracking for variance analysis.

Auto-Detect Strategy: Intelligent layer that recommends the optimal statistical path (PCA, Regression, etc.) based on data architecture.

High-Fidelity Visuals: Dynamic Seaborn and Matplotlib plots refactored for multi-hue dataset distinction.

Professional Export: Server-side document generation into .docx format, complete with native tables and embedded plots.

Secure Lifecycle: Full Google OAuth 2.0 integration.

## 📖 How It Works
BT IntelliStat is designed for a streamlined research workflow:

Select Volume: Choose to analyze a single file or perform a comparative synthesis of up to 3 datasets.

Upload & Validate: Drop your .csv or .xlsx files into the secure upload zone. The system automatically verifies headers and cleans empty rows for mathematical consistency.

Configure Methodology: Select between AI Auto-Detect (which chooses the best statistical path) or manually select modules like K-Means, Linear Regression, or PCA.

Execute & Synthesize: The engine processes the data through the Python scientific stack and triggers the Triple-Tier AI Failover to generate qualitative insights.

Export Results: Download a high-fidelity .docx report containing the executive summary, data tables, and high-resolution visualizations formatted for academic submission.

## 🛠️ Tech Stack
Backend: Python 3.10+ / Flask / Gunicorn

AI Ecosystem: Google Gemini 2.5 (Flash & Lite) + Groq (Llama 3.3)

Scientific Stack: Pandas, NumPy, Scikit-Learn

Visualization: Matplotlib & Seaborn

Infrastructure: Render with Persistent Volume & Custom DNS

## 📐 Implementation Logic
To maintain academic and technical integrity, the system implements:

Memory Management: Strategic row-sampling for massive files to maintain low-latency response times.

Source Attribution: Every merged row maintains a Source_File identifier to ensure data provenance.

Failover Transparency: Real-time logging of model failovers to monitor API health.

🛡️ License & Copyright
© 2026 BT IntelliStat. Developed by Tvisha Majithia for Senior Research and Data Science Applications.
