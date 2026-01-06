import os
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
from authlib.integrations.flask_client import OAuth
from groq import Groq
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import re
from datetime import datetime
from googleapiclient.http import MediaIoBaseUpload

app = Flask(__name__)

# --- CONFIGURATION (Use Environment Variables) ---
app.secret_key = os.environ.get("SECRET_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Groq client with error handling
try:
    ai_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    ai_client = None

oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile https://www.googleapis.com/auth/drive.appdata https://www.googleapis.com/auth/drive.file'
},
)

# --- RESEARCH ENGINE ---

def format_tables_in_html(html_content):
    """Convert text-based tables in HTML to proper HTML tables"""
    if '|' in html_content and '---' in html_content:
        lines = html_content.split('\n')
        table_lines = []
        in_table = False
        
        for line in lines:
            if '|' in line and line.strip():
                in_table = True
                table_lines.append(line)
            elif in_table and not line.strip():
                if table_lines:
                    table_html = '<table style="width:100%; border-collapse: collapse; margin: 1rem 0;">'
                    
                    if table_lines:
                        headers = [cell.strip() for cell in table_lines[0].split('|') if cell.strip()]
                        table_html += '<thead><tr>'
                        for header in headers:
                            table_html += f'<th style="background: rgba(102, 126, 234, 0.2); color: white; padding: 12px; text-align: left; font-weight: 600;">{header}</th>'
                        table_html += '</tr></thead><tbody>'
                    
                    for row_line in table_lines[2:]:
                        cells = [cell.strip() for cell in row_line.split('|') if cell.strip()]
                        if cells:
                            table_html += '<tr>'
                            for cell in cells:
                                table_html += f'<td style="padding: 10px 12px; border-bottom: 1px solid rgba(255, 255, 255, 0.05);">{cell}</td>'
                            table_html += '</tr>'
                    
                    table_html += '</tbody></table>'
                    html_content = html_content.replace('\n'.join(table_lines), table_html)
                
                table_lines = []
                in_table = False
    
    return html_content

def get_professional_insight(method_name, stats_data, df_summary, is_comparison=False):
    """Get AI insights with error handling"""
    if not ai_client:
        return "<p><b>Error:</b> AI service is not available. Please check your GROQ API key.</p>"
    
    comp_task = "Compare both results. Explicitly state which is more accurate/useful and WHY." if is_comparison else ""
    prompt = f"""
    ROLE: Senior Research Data Scientist.
    METHOD: {method_name}
    STATS: {stats_data}
    CONTEXT: {df_summary}
    TASK: Provide a professional, extremely detailed and in-depth interpretation of the results.
    FORMATTING RULES:
    1. Output ONLY valid HTML. Do NOT use Markdown symbols like '**', '###', or '==='.
    2. Use <h3> for section titles.
    3. Use <b> for key metrics and <ul>/<li> for findings.
    4. If presenting tabular data, use proper HTML <table> tags with <thead>, <tbody>, <tr>, <th>, and <td>.
    5. Ensure all text is wrapped in <p> tags.
    {comp_task}"""
    
    try:
        completion = ai_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=2000
        )
        
        insight = completion.choices[0].message.content
        insight = re.sub(r'={3,}', '', insight)
        insight = re.sub(r'-{3,}', '', insight)
        insight = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', insight)
        return format_tables_in_html(insight)
        return insight
    except Exception as e:
        error_msg = str(e)
        if "invalid_api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            return f"<p><b>Authentication Error:</b> Invalid or expired Groq API key. Please check your API key configuration.</p><p><small>Error: {error_msg}</small></p>"
        else:
            return f"<p><b>Error generating insights:</b> {error_msg}</p>"

def create_custom_graph(df, graph_type):
    """Create specific graph types for custom analysis"""
    num_df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    if num_df.empty:
        return None
    
    imputer = SimpleImputer(strategy='mean')
    clean_num = pd.DataFrame(imputer.fit_transform(num_df), columns=num_df.columns)
    
    plt.figure(figsize=(12, 8))
    
    try:
        if graph_type == 'bar':
            if clean_num.shape[1] >= 2:
                clean_num.iloc[:, :2].plot(kind='bar', ax=plt.gca())
                plt.title('Bar Chart Comparison', fontsize=14, fontweight='bold')
                plt.xlabel(clean_num.columns[0])
                plt.ylabel('Values')
                plt.xticks(rotation=45)
                plt.legend()
                plt.tight_layout()
        
        elif graph_type == 'line':
            for col in clean_num.columns[:3]:
                plt.plot(clean_num.index, clean_num[col], marker='o', label=col)
            plt.title('Line Chart Trends', fontsize=14, fontweight='bold')
            plt.xlabel('Index')
            plt.ylabel('Values')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        
        elif graph_type == 'pie':
            if clean_num.shape[1] >= 1:
                values = clean_num.iloc[:, 0].value_counts().head(10)
                plt.pie(values, labels=values.index, autopct='%1.1f%%', startangle=90)
                plt.title('Pie Chart Distribution', fontsize=14, fontweight='bold')
                plt.axis('equal')
        
        elif graph_type == 'scatter':
            if clean_num.shape[1] >= 2:
                plt.scatter(clean_num.iloc[:, 0], clean_num.iloc[:, 1], alpha=0.6, s=50)
                plt.title('Scatter Plot', fontsize=14, fontweight='bold')
                plt.xlabel(clean_num.columns[0])
                plt.ylabel(clean_num.columns[1])
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
        
        elif graph_type == 'histogram':
            clean_num.iloc[:, 0].plot(kind='hist', bins=30, edgecolor='black', alpha=0.7)
            plt.title('Histogram Distribution', fontsize=14, fontweight='bold')
            plt.xlabel(clean_num.columns[0])
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        
        elif graph_type == 'heatmap':
            sns.heatmap(clean_num.corr(), annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
            plt.tight_layout()
        
        elif graph_type == 'box':
            clean_num.boxplot(figsize=(12, 8))
            plt.title('Box & Whisker Plot', fontsize=14, fontweight='bold')
            plt.ylabel('Values')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        
        elif graph_type == 'waterfall':
            if clean_num.shape[1] >= 1:
                values = clean_num.iloc[:10, 0].values
                cumulative = np.cumsum(values)
                plt.bar(range(len(values)), values, alpha=0.7)
                plt.plot(range(len(values)), cumulative, 'r-o', linewidth=2, markersize=8, label='Cumulative')
                plt.title('Waterfall Chart', fontsize=14, fontweight='bold')
                plt.xlabel('Categories')
                plt.ylabel('Values')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return plot_url
        
    except Exception as e:
        plt.close()
        return None

def run_analysis(df, method, custom_query=None, custom_graph_type=None):
    """Executes methodology and returns (Base64_Image_String, Stats_String, Method_Name, Has_Table)."""
    num_df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    if num_df.empty:
        return None, "Error: No numerical data found.", method, False
    
    imputer = SimpleImputer(strategy='mean')
    clean_num = pd.DataFrame(imputer.fit_transform(num_df), columns=num_df.columns)
    
    plt.figure(figsize=(12, 8))
    stats_str, final_name = "", method.replace("_", " ").title()
    has_table = False

    try:
        if method == "auto_detect":
            summary = clean_num.describe().to_string()
            
            if ai_client:
                try:
                    ai_res = ai_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role":"user", "content": f"""
                        Analyze this data summary: {summary}
                        Identify the single most mathematically precise methodology for this data.
                        Return ONLY the name."""}]
                    )
                    detected_name = ai_res.choices[0].message.content.strip()
                except:
                    detected_name = "Statistical Summary"
            else:
                detected_name = "Statistical Summary"
            
            has_table = True
            return None, summary, f"AI Recommended: {detected_name}", has_table

        elif method == "correlation":
            sns.heatmap(clean_num.corr(), annot=True, cmap='RdBu_r', center=0, 
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title("Correlation Heatmap", fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            stats_str = clean_num.corr().to_string()

        elif method == "kmeans" and clean_num.shape[1] >= 2:
            model = KMeans(n_clusters=3, n_init='auto', random_state=42).fit(clean_num)
            clean_num['Cluster'] = model.labels_
            sns.scatterplot(data=clean_num, x=clean_num.columns[0], y=clean_num.columns[1], 
                          hue='Cluster', palette='viridis', s=100, alpha=0.7)
            plt.title("K-Means Segmentation", fontsize=14, fontweight='bold', pad=20)
            plt.xlabel(clean_num.columns[0], fontsize=12)
            plt.ylabel(clean_num.columns[1], fontsize=12)
            plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            stats_str = f"Inertia: {model.inertia_:.2f}"

        elif method == "random_forest" and clean_num.shape[1] >= 2:
            X = clean_num.drop(clean_num.columns[0], axis=1)
            y = clean_num.iloc[:, 0]
            rf = RandomForestRegressor(n_estimators=100).fit(X, y)
            imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
            imp.plot(kind='barh', color='steelblue', edgecolor='black')
            plt.title("Feature Importance", fontsize=14, fontweight='bold', pad=20)
            plt.xlabel('Importance Score', fontsize=12)
            plt.tight_layout()
            stats_str = imp.to_string()
            has_table = True

        elif method == "regression" and clean_num.shape[1] >= 2:
            sns.regplot(data=clean_num, x=clean_num.columns[0], y=clean_num.columns[-1], 
                       scatter_kws={'alpha':0.5, 's':50}, line_kws={'color':'red', 'linewidth':2})
            plt.title("Linear Regression", fontsize=14, fontweight='bold', pad=20)
            plt.xlabel(clean_num.columns[0], fontsize=12)
            plt.ylabel(clean_num.columns[-1], fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            stats_str = "Regression Trendline calculated."

        elif method == "anova":
            cat_df = df.select_dtypes(exclude=[np.number])
            if not cat_df.empty:
                sns.boxplot(x=df[cat_df.columns[0]], y=clean_num[clean_num.columns[0]], palette='Set2')
                plt.title("ANOVA: Group Variance", fontsize=14, fontweight='bold', pad=20)
                plt.xlabel(cat_df.columns[0], fontsize=12)
                plt.ylabel(clean_num.columns[0], fontsize=12)
                plt.xticks(rotation=45)
                plt.tight_layout()
                stats_str = "Group differences visualized via Boxplot."
            else:
                sns.boxenplot(data=clean_num, palette='muted')
                plt.title("Numeric Distributions", fontsize=14, fontweight='bold', pad=20)
                plt.xticks(rotation=45)
                plt.tight_layout()
                stats_str = "No categorical data found; displaying numeric distributions."

        elif method == "pca" and clean_num.shape[1] >= 2:
            pca = PCA(n_components=2)
            components = pca.fit_transform(clean_num)
            plt.scatter(components[:, 0], components[:, 1], c='blue', alpha=0.6, s=50, edgecolors='black')
            plt.title("PCA Dimensionality Reduction", fontsize=14, fontweight='bold', pad=20)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            stats_str = f"Explained Variance: {pca.explained_variance_ratio_}"
            has_table = True

        elif method == "ttest" and clean_num.shape[1] >= 1:
            sns.kdeplot(clean_num.iloc[:, 0], fill=True, color='purple', alpha=0.6)
            plt.title("T-Test Distribution Analysis", fontsize=14, fontweight='bold', pad=20)
            plt.xlabel(clean_num.columns[0], fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            stats_str = f"Mean of target: {clean_num.iloc[:,0].mean():.2f}"
            has_table = True

        elif method == "distribution":
            sns.histplot(clean_num.iloc[:, 0], kde=True, color='green', bins=30, edgecolor='black')
            plt.title("Frequency Distribution", fontsize=14, fontweight='bold', pad=20)
            plt.xlabel(clean_num.columns[0], fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            stats_str = f"Mean: {clean_num.iloc[:,0].mean():.2f}, SD: {clean_num.iloc[:,0].std():.2f}"
            has_table = True

        elif method == "timeseries":
            plt.plot(clean_num.iloc[:, 0], marker='o', linestyle='-', linewidth=2, markersize=5)
            plt.title("Temporal Trend Analysis", fontsize=14, fontweight='bold', pad=20)
            plt.xlabel('Time Index', fontsize=12)
            plt.ylabel(clean_num.columns[0], fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            stats_str = "Time-series data plotted sequentially."

        elif method == "other":
            if custom_graph_type:
                plt.close()
                plot_url = create_custom_graph(df, custom_graph_type)
                final_name = f"Custom: {custom_graph_type.title()} View"
                stats_str = f"Custom {custom_graph_type} visualization created based on: {custom_query}"
                return plot_url, stats_str, final_name, False
            else:
                if clean_num.shape[1] >= 2:
                    pairplot_data = clean_num.iloc[:, :4]
                    g = sns.pairplot(pairplot_data, diag_kind='kde', plot_kws={'alpha':0.6, 's':50})
                    g.fig.suptitle(f"Custom Pairplot: {custom_query[:30]}", y=1.01, fontsize=14, fontweight='bold')
                    plt.tight_layout()
                else:
                    clean_num.iloc[:, 0].plot(kind='hist', bins=30, edgecolor='black', alpha=0.7)
                    plt.title(f"Custom View: {custom_query[:30]}", fontsize=14, fontweight='bold', pad=20)
                stats_str = f"Manual query processed: {custom_query}"

        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=150, facecolor='white')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return plot_url, stats_str, final_name, has_table

    except Exception as e:
        plt.close()
        return None, f"Error: {str(e)}", final_name, False

def save_analysis_to_drive(filename, method, html_content):
    if 'google_token' not in session: return
    try:
        creds = Credentials(token=session['google_token']['access_token'])
        service = build('drive', 'v3', credentials=creds)

        analysis_data = {
            "filename": filename,
            "method": method,
            "timestamp": datetime.now().isoformat(),
            "html": html_content
        }
        
        file_metadata = {
            'name': f"BT_{filename}_{method}.json",
            'parents': ['appDataFolder'] # Save in hidden folder
        }
        
        media = MediaIoBaseUpload(
            io.BytesIO(json.dumps(analysis_data).encode('utf-8')),
            mimetype='application/json'
        )
        
        service.files().create(body=file_metadata, media_body=media).execute()
    except Exception as e:
        print(f"Drive Error: {e}")

# --- ROUTES ---

@app.route("/")
def index():
    return render_template("login.html")

@app.route("/login/google")
def login_google():
    return google.authorize_redirect(url_for('google_auth', _external=True))

@app.route("/google/auth")
def google_auth():
    token = google.authorize_access_token()
    user = google.get('https://openidconnect.googleapis.com/v1/userinfo').json()
    session['google_token'], session['user_name'] = token, user.get('name')
    return redirect(url_for('dashboard'))

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if 'google_token' not in session: 
        return redirect(url_for('index'))
    
    results = []
    current_filename = "Unknown File"

    if request.method == "POST":
        file = request.files.get("file")
        mode = request.form.get("mode")
        m1, m2 = request.form.get("method"), request.form.get("method2")
        custom_txt = request.form.get("custom_query")
        custom_graph = request.form.get("custom_graph_type")

        # 1. Handle File Upload or Retrieval with Google Drive Check
        if file and file.filename != '':
            current_filename = file.filename
            local_path = os.path.join(UPLOAD_FOLDER, current_filename)
            file.save(local_path)  # Always save locally for current processing
            session['last_file'] = local_path

            try:
                # Initialize Drive Service
                creds = Credentials(token=session['google_token']['access_token'])
                drive_service = build('drive', 'v3', credentials=creds)
                
                # Check if file exists on Drive
                query = f"name = '{current_filename}' and trashed = false"
                drive_response = drive_service.files().list(q=query, fields="files(id, name)").execute()
                existing_files = drive_response.get('files', [])

                if not existing_files:
                    # File not on Drive -> Upload it
                    file_metadata = {'name': current_filename}
                    media = MediaFileUpload(local_path, mimetype='text/csv')
                    drive_service.files().create(body=file_metadata, media_body=media).execute()
                    print(f"Uploaded {current_filename} to Drive.")
                else:
                    print(f"{current_filename} already exists on Drive. Skipping upload.")
            
            except Exception as e:
                print(f"Drive Check/Upload Error: {e}")

        elif session.get('last_file'):
            current_filename = os.path.basename(session.get('last_file'))

        # 2. Perform Analysis
        if session.get('last_file'):
            try:
                file_path = session['last_file']
                file_ext = os.path.splitext(file_path)[1].lower()
                
                if file_ext == '.csv':
                    df = pd.read_csv(file_path, encoding='latin1')
                elif file_ext in ['.xlsx', '.xls']:
                    df = pd.read_excel(file_path)
                else:
                    raise ValueError(f"Unsupported file format: {file_ext}")
                
                # First analysis
                p1, s1, n1, t1 = run_analysis(df, m1, custom_txt, custom_graph)
                i1 = get_professional_insight(n1, s1, df.describe().to_string())
                results.append({'plot': p1, 'insight': i1, 'name': n1, 'has_table': t1})

                # Comparative analysis
                if mode == "comparative" and m2:
                    p2, s2, n2, t2 = run_analysis(df, m2)
                    i2 = get_professional_insight(f"Compare: {n1} vs {n2}", f"M1: {s1} | M2: {s2}", df.describe().to_string(), True)
                    results.append({'plot': p2, 'insight': i2, 'name': n2, 'has_table': t2})
            
            except Exception as e:
                results = [{'plot': None, 'insight': f'<p><b>Error processing file:</b> {str(e)}</p>', 'name': 'Error', 'has_table': False}]

            # 3. Save Analysis History to appDataFolder
            if results and results[0]['name'] != 'Error':
                save_analysis_to_drive(current_filename, results[0]['name'], render_template("dashboard_partial.html", results=results, mode=mode))
            
            # 4. Return AJAX Response
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    "status": "success",
                    "filename": current_filename,
                    "method": results[0]['name'] if results else "Analysis",
                    "html": render_template("dashboard_partial.html", results=results, mode=mode)
                })

    return render_template("dashboard.html", user=session.get('user_name'))
    return render_template("dashboard.html",user=session.get('user_name'),current_year=datetime.now().year)

@app.route("/get_cloud_history")
def get_cloud_history():
    if 'google_token' not in session: 
        return jsonify({"error": "Not authenticated"}), 401
    
    try:
        creds = Credentials(token=session['google_token']['access_token'])
        service = build('drive', 'v3', credentials=creds)
        
        # We MUST specify spaces='appDataFolder' to see these files
        results = service.files().list(
            spaces='appDataFolder',
            fields="files(id, name, createdTime)",
            pageSize=15,
            orderBy='createdTime desc'
        ).execute()
        
        files = results.get('files', [])
        return jsonify(files)
    except Exception as e:
        print(f"Drive history error: {str(e)}")
        return jsonify({"error": f"Failed to load history: {str(e)}"}), 500

@app.route("/get_analysis_detail/<file_id>")
def get_analysis_detail(file_id):
    creds = Credentials(token=session['google_token']['access_token'])
    service = build('drive', 'v3', credentials=creds)
    
    content = service.files().get_media(fileId=file_id).execute()
    return jsonify(json.loads(content))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
