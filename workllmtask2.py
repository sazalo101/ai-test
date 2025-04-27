import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import sqlite3
from datetime import datetime
import uuid
import google.generativeai as genai
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ReviewProcessor:
    def __init__(self, model_name='all-MiniLM-L6-v2', gemini_api_key=None):
        """Initialize the review processor with a sentence transformer model and Gemini API."""
        self.model = SentenceTransformer(model_name)
        self.conn = sqlite3.connect('reviews.db')
        self.create_database()
        self.classifier = None
        
        # Initialize Gemini API
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
        else:
            raise ValueError("Gemini API key is required")

    def create_database(self):
        """Create SQLite database tables and ensure predicted_sentiment column exists."""
        cursor = self.conn.cursor()
        
        # Create reviews table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reviews (
                review_id TEXT PRIMARY KEY,
                product TEXT,
                category TEXT,
                rating INTEGER,
                review_text TEXT,
                feature_mentioned TEXT,
                attribute_mentioned TEXT,
                date TEXT,
                sentiment TEXT
            )
        ''')
        
        # Check if predicted_sentiment column exists and add it if not
        cursor.execute("PRAGMA table_info(reviews)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'predicted_sentiment' not in columns:
            cursor.execute('''
                ALTER TABLE reviews
                ADD COLUMN predicted_sentiment TEXT
            ''')
        
        # Create embeddings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                review_id TEXT PRIMARY KEY,
                embedding BLOB,
                FOREIGN KEY (review_id) REFERENCES reviews (review_id)
            )
        ''')
        
        self.conn.commit()

    def load_data(self, file_path):
        """Load review data from CSV or JSONL file."""
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.jsonl'):
            df = pd.read_json(file_path, lines=True)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSONL.")
        return df

    def process_reviews(self, df):
        """Process reviews, generate embeddings, and store in database."""
        cursor = self.conn.cursor()
        
        # Generate embeddings for review_text
        review_texts = df['review_text'].tolist()
        embeddings = self.model.encode(review_texts, show_progress_bar=True)
        
        # Store reviews and embeddings
        for _, row in df.iterrows():
            review_id = str(row['review_id'])
            
            # Store review data
            cursor.execute('''
                INSERT OR REPLACE INTO reviews (
                    review_id, product, category, rating, review_text,
                    feature_mentioned, attribute_mentioned, date, sentiment
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                review_id,
                row['product'],
                row['category'],
                row['rating'],
                row['review_text'],
                row['feature_mentioned'],
                row['attribute_mentioned'],
                row['date'],
                row['sentiment']
            ))
            
            # Store embedding
            embedding_blob = embeddings[_].tobytes()
            cursor.execute('''
                INSERT OR REPLACE INTO embeddings (review_id, embedding)
                VALUES (?, ?)
            ''', (review_id, embedding_blob))
        
        self.conn.commit()

    def train_sentiment_classifier(self):
        """Train a sentiment classifier using review embeddings and labeled sentiments."""
        cursor = self.conn.cursor()
        
        # Fetch review data and embeddings
        cursor.execute('SELECT review_id, sentiment FROM reviews')
        review_data = cursor.fetchall()
        cursor.execute('SELECT review_id, embedding FROM embeddings')
        embedding_data = cursor.fetchall()
        
        # Prepare data for classification
        review_dict = {r[0]: r[1] for r in review_data}
        X = []
        y = []
        for review_id, embedding_blob in embedding_data:
            if review_id in review_dict:
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                X.append(embedding)
                y.append(review_dict[review_id])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest Classifier
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X_train, y_train)
        
        # Evaluate classifier
        y_pred = self.classifier.predict(X_test)
        print("\nSentiment Classifier Performance:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        
        # Store predicted sentiments
        all_predictions = self.classifier.predict(X)
        for i, (review_id, _) in enumerate(embedding_data):
            cursor.execute('''
                UPDATE reviews
                SET predicted_sentiment = ?
                WHERE review_id = ?
            ''', (all_predictions[i], review_id))
        
        self.conn.commit()

    def get_similar_reviews(self, query, top_k=5):
        """Retrieve reviews semantically similar to the query."""
        cursor = self.conn.cursor()
        
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Fetch all embeddings
        cursor.execute('SELECT review_id, embedding FROM embeddings')
        results = cursor.fetchall()
        
        similarities = []
        for review_id, embedding_blob in results:
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                embedding.reshape(1, -1)
            )[0][0]
            similarities.append((review_id, similarity))
        
        # Sort by similarity and get top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_reviews = similarities[:top_k]
        
        # Fetch review details
        similar_reviews = []
        for review_id, similarity in top_reviews:
            cursor.execute('SELECT * FROM reviews WHERE review_id = ?', (review_id,))
            review = cursor.fetchone()
            similar_reviews.append({
                'review_id': review[0],
                'product': review[1],
                'category': review[2],
                'rating': review[3],
                'review_text': review[4],
                'feature_mentioned': review[5],
                'attribute_mentioned': review[6],
                'date': review[7],
                'sentiment': review[8],
                'predicted_sentiment': review[9],
                'similarity': similarity
            })
        
        return similar_reviews

    def generate_category_summary(self):
        """Generate concise summaries of product performance by category using Gemini."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT DISTINCT category FROM reviews')
        categories = [row[0] for row in cursor.fetchall()]
        
        summaries = {}
        for category in categories:
            cursor.execute('''
                SELECT product, rating, review_text, sentiment, predicted_sentiment
                FROM reviews
                WHERE category = ?
                LIMIT 50
            ''', (category,))
            reviews = cursor.fetchall()
            
            # Prepare review data for Gemini
            review_texts = [
                f"Product: {r[0]}, Rating: {r[1]}, Review: {r[2]}, Sentiment: {r[3]}, Predicted: {r[4]}"
                for r in reviews
            ]
            prompt = f"""
            Generate a concise summary (100-150 words) of product performance for the {category} category
            based on the following reviews. Highlight average rating, common features mentioned,
            overall sentiment, and compare labeled vs predicted sentiments:
            {chr(10).join(review_texts)}
            """
            
            response = self.gemini_model.generate_content(prompt)
            summaries[category] = response.text
        
        return summaries

    def answer_product_question(self, question):
        """Answer specific questions about products using Gemini and review data."""
        # Find relevant reviews using semantic search
        similar_reviews = self.get_similar_reviews(question, top_k=10)
        
        # Prepare context for Gemini
        context = [
            f"Product: {r['product']}, Category: {r['category']}, Rating: {r['rating']}, "
            f"Review: {r['review_text']}, Feature: {r['feature_mentioned']}, "
            f"Attribute: {r['attribute_mentioned']}, Sentiment: {r['sentiment']}, "
            f"Predicted Sentiment: {r['predicted_sentiment']}"
            for r in similar_reviews
        ]
        
        prompt = f"""
        Based on the following review data, answer the question: '{question}'
        Provide a clear, concise answer (100-200 words) supported by the review data.
        Review data:
        {chr(10).join(context)}
        """
        
        response = self.gemini_model.generate_content(prompt)
        return response.text

    def identify_category_trends(self):
        """Identify common issues and praised features across product categories using Gemini."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT DISTINCT category FROM reviews')
        categories = [row[0] for row in cursor.fetchall()]
        
        trends = {}
        for category in categories:
            cursor.execute('''
                SELECT review_text, feature_mentioned, attribute_mentioned, sentiment, predicted_sentiment
                FROM reviews
                WHERE category = ?
                LIMIT 50
            ''', (category,))
            reviews = cursor.fetchall()
            
            # Prepare review data for Gemini
            review_texts = [
                f"Review: {r[0]}, Feature: {r[1]}, Attribute: {r[2]}, Sentiment: {r[3]}, Predicted: {r[4]}"
                for r in reviews
            ]
            prompt = f"""
            Analyze the following reviews for the {category} category and identify:
            1. Common issues (negative feedback)
            2. Praised features (positive feedback)
            Provide a concise summary (150-200 words) for each category, considering both labeled and predicted sentiments.
            Review data:
            {chr(10).join(review_texts)}
            """
            
            response = self.gemini_model.generate_content(prompt)
            trends[category] = response.text
        
        return trends

    def create_sentiment_dashboard(self, output_file='sentiment_dashboard.html'):
        """Create a visualization dashboard showing sentiment trends over time and across categories."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT date, category, sentiment, predicted_sentiment FROM reviews')
        data = cursor.fetchall()
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=['date', 'category', 'sentiment', 'predicted_sentiment'])
        df['date'] = pd.to_datetime(df['date'])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Sentiment Distribution by Category",
                "Predicted Sentiment Distribution",
                "Sentiment Trend Over Time",
                "Sentiment Comparison"
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                  [{"type": "scatter"}, {"type": "heatmap"}]]
        )
        
        # Plot 1: Sentiment Distribution by Category
        sentiment_counts = df.groupby(['category', 'sentiment']).size().unstack(fill_value=0)
        for sentiment in sentiment_counts.columns:
            fig.add_trace(
                go.Bar(
                    x=sentiment_counts.index,
                    y=sentiment_counts[sentiment],
                    name=sentiment
                ),
                row=1, col=1
            )
        
        # Plot 2: Predicted Sentiment Distribution
        pred_sentiment_counts = df.groupby(['category', 'predicted_sentiment']).size().unstack(fill_value=0)
        for sentiment in pred_sentiment_counts.columns:
            fig.add_trace(
                go.Bar(
                    x=pred_sentiment_counts.index,
                    y=pred_sentiment_counts[sentiment],
                    name=f"Pred_{sentiment}"
                ),
                row=1, col=2
            )
        
        # Plot 3: Sentiment Trend Over Time
        df['month'] = df['date'].dt.to_period('M').astype(str)
        time_trend = df.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
        for sentiment in time_trend.columns:
            fig.add_trace(
                go.Scatter(
                    x=time_trend.index,
                    y=time_trend[sentiment],
                    mode='lines+markers',
                    name=sentiment
                ),
                row=2, col=1
            )
        
        # Plot 4: Sentiment Comparison Heatmap
        confusion_matrix = pd.crosstab(df['sentiment'], df['predicted_sentiment'])
        fig.add_trace(
            go.Heatmap(
                z=confusion_matrix.values,
                x=confusion_matrix.columns,
                y=confusion_matrix.index,
                colorscale='Viridis'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Sentiment Analysis Dashboard",
            height=800,
            showlegend=True,
            barmode='stack'
        )
        
        # Save to HTML
        fig.write_html(output_file)
        print(f"\nSentiment dashboard saved to {output_file}")

    def close(self):
        """Close database connection."""
        self.conn.close()

def main():
    # Example usage
    processor = ReviewProcessor(gemini_api_key=os.getenv('GEMINI_API_KEY'))
    
    try:
        # Load and process data
        df = processor.load_data('product_reviews.csv')
        processor.process_reviews(df)
        
        # Train sentiment classifier
        processor.train_sentiment_classifier()
        
        # Generate category summaries
        summaries = processor.generate_category_summary()
        print("\nCategory Summaries:")
        for category, summary in summaries.items():
            print(f"\n{category}:\n{summary}")
        
        # Example Q&A
        question = "What are the main complaints about smartphone battery life?"
        answer = processor.answer_product_question(question)
        print(f"\nQ&A Example:\nQuestion: {question}\nAnswer: {answer}")
        
        # Identify trends
        trends = processor.identify_category_trends()
        print("\nCategory Trends:")
        for category, trend in trends.items():
            print(f"\n{category}:\n{trend}")
        
        # Create visualization dashboard
        processor.create_sentiment_dashboard()
            
    finally:
        processor.close()

if __name__ == '__main__':
    main()