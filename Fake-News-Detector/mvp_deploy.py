#!/usr/bin/env python3
"""
Quick Deployment Script for Fake News Detection API

Usage:
    python mvp_deploy.py --export-model
    python mvp_deploy.py --test-local
    python mvp_deploy.py --deploy heroku
"""

import os
import sys
import subprocess
import argparse
import json
import joblib
from pathlib import Path
from datetime import datetime


class QuickDeployment:
    def __init__(self):
        self.project_root = Path.cwd()
        self.model_dir = self.project_root / "production_models" / "1.0.0"
        self.static_dir = self.project_root / "static"

    def export_model(self, pipeline_results=None):
        """Export trained model for production"""
        print("🚀 Exporting model for production...")

        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)

        if pipeline_results:
            # If results provided, export from there
            model = pipeline_results.get('ensemble_model')
            vectorizer = pipeline_results.get('vectorizer')
        else:
            # Load from pipeline results
            print("⚠️  No pipeline results provided. Please provide model and vectorizer.")
            print("   Usage: deploy.export_model(pipeline_results)")
            return False

        # Save model artifacts
        print("   Saving model...")
        joblib.dump(model, self.model_dir / "model.pkl")

        print("   Saving vectorizer...")
        joblib.dump(vectorizer, self.model_dir / "vectorizer.pkl")

        # Save metadata
        metadata = {
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "metrics": {
                "f1_score": 0.596,
                "accuracy": 0.617,
                "cv_mean": 0.588
            },
            "model_type": "VotingClassifier",
            "features": "TF-IDF (5000 features)",
            "dataset": "LIAR (12,791 samples)"
        }

        with open(self.model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Calculate sizes
        model_size = (self.model_dir / "model.pkl").stat().st_size / 1024 / 1024
        vec_size = (self.model_dir / "vectorizer.pkl").stat().st_size / 1024 / 1024

        print(f"\n✅ Model exported successfully!")
        print(f"   Location: {self.model_dir}")
        print(f"   Model size: {model_size:.2f} MB")
        print(f"   Vectorizer size: {vec_size:.2f} MB")

        return True

    def create_api_files(self):
        """Create necessary API files if they don't exist"""
        print("\n📁 Creating API files...")

        files_created = []

        # Create requirements.txt
        requirements = """fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
scikit-learn==1.3.0
joblib==1.3.2
numpy==1.24.0
"""

        req_file = self.project_root / "requirements.txt"
        if not req_file.exists():
            req_file.write_text(requirements)
            files_created.append("requirements.txt")

        # Create .gitignore
        gitignore = """__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
*.log
.DS_Store
liar_analysis_results/
data/
*.pkl
!production_models/**/*.pkl
"""

        git_file = self.project_root / ".gitignore"
        if not git_file.exists():
            git_file.write_text(gitignore)
            files_created.append(".gitignore")

        # Create Procfile for Heroku
        procfile = "web: uvicorn api:app --host 0.0.0.0 --port $PORT"

        proc_file = self.project_root / "Procfile"
        if not proc_file.exists():
            proc_file.write_text(procfile)
            files_created.append("Procfile")

        # Create runtime.txt for Heroku
        runtime = "python-3.11.0"

        runtime_file = self.project_root / "runtime.txt"
        if not runtime_file.exists():
            runtime_file.write_text(runtime)
            files_created.append("runtime.txt")

        if files_created:
            print(f"   Created: {', '.join(files_created)}")
        else:
            print("   All files already exist")

        return True

    def test_local(self):
        """Test the API locally"""
        print("\n🧪 Testing API locally...")

        # Check if model exists
        if not (self.model_dir / "model.pkl").exists():
            print("❌ Model not found! Run --export-model first")
            return False

        # Check if api.py exists
        if not (self.project_root / "api.py").exists():
            print("❌ api.py not found! Please create the API file")
            return False

        print("   Starting server on http://localhost:8000")
        print("   Press Ctrl+C to stop")
        print("\n   Testing endpoints:")
        print("   - Health: http://localhost:8000/health")
        print("   - Docs: http://localhost:8000/docs")
        print("   - Frontend: http://localhost:8000\n")

        try:
            subprocess.run([
                sys.executable, "-m", "uvicorn",
                "api:app", "--reload", "--port", "8000"
            ])
        except KeyboardInterrupt:
            print("\n\n✅ Server stopped")

        return True

    def build_docker(self):
        """Build Docker image"""
        print("\n🐳 Building Docker image...")

        # Check if Dockerfile exists
        dockerfile = self.project_root / "Dockerfile"
        if not dockerfile.exists():
            print("❌ Dockerfile not found! Please create it first")
            return False

        try:
            # Build image
            print("   Building image (this may take a few minutes)...")
            subprocess.run([
                "docker", "build",
                "-t", "fake-news-api:1.0.0",
                "."
            ], check=True)

            print("\n✅ Docker image built successfully!")
            print("   Image: fake-news-api:1.0.0")
            print("\n   To run:")
            print("   docker run -d -p 8000:8000 fake-news-api:1.0.0")

            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Docker build failed: {e}")
            return False
        except FileNotFoundError:
            print("❌ Docker not installed. Install from https://docker.com")
            return False

    def deploy_heroku(self):
        """Deploy to Heroku"""
        print("\n☁️  Deploying to Heroku...")

        try:
            # Check if heroku CLI is installed
            subprocess.run(["heroku", "--version"],
                           capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Heroku CLI not installed")
            print("   Install from: https://devcenter.heroku.com/articles/heroku-cli")
            return False

        # Check if git is initialized
        if not (self.project_root / ".git").exists():
            print("   Initializing git repository...")
            subprocess.run(["git", "init"], check=True)
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], check=True)

        # Create Heroku app
        print("   Creating Heroku app...")
        app_name = f"fake-news-{datetime.now().strftime('%Y%m%d%H%M')}"

        try:
            result = subprocess.run([
                "heroku", "create", app_name
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print(f"   App created: {app_name}")
            else:
                print("   App already exists or creation failed")
        except subprocess.CalledProcessError:
            print("   Using existing app")

        # Deploy
        print("   Deploying to Heroku (this may take several minutes)...")
        try:
            subprocess.run([
                "git", "push", "heroku", "main"
            ], check=True)

            print("\n✅ Deployed successfully!")
            print(f"   URL: https://{app_name}.herokuapp.com")

            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Deployment failed: {e}")
            print("   Try: heroku logs --tail")
            return False

    def deploy_cloud_run(self, project_id):
        """Deploy to Google Cloud Run"""
        print("\n☁️  Deploying to Google Cloud Run...")

        if not project_id:
            print("❌ Please provide Google Cloud project ID")
            print("   Usage: python mvp_deploy.py --deploy cloudrun --project YOUR_PROJECT_ID")
            return False

        try:
            # Check if gcloud is installed
            subprocess.run(["gcloud", "--version"],
                           capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ gcloud CLI not installed")
            print("   Install from: https://cloud.google.com/sdk/docs/install")
            return False

        # Set project
        print(f"   Setting project: {project_id}")
        subprocess.run(["gcloud", "config", "set", "project", project_id], check=True)

        # Build and push
        print("   Building and pushing image...")
        image_name = f"gcr.io/{project_id}/fake-news-api"

        try:
            subprocess.run([
                "gcloud", "builds", "submit",
                "--tag", image_name
            ], check=True)

            # Deploy
            print("   Deploying to Cloud Run...")
            subprocess.run([
                "gcloud", "run", "deploy", "fake-news-api",
                "--image", image_name,
                "--platform", "managed",
                "--region", "us-central1",
                "--allow-unauthenticated",
                "--memory", "1Gi"
            ], check=True)

            print("\n✅ Deployed successfully!")
            print("   Check Cloud Console for URL")

            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Deployment failed: {e}")
            return False

    def full_setup(self):
        """Run complete setup"""
        print("🚀 Running full setup...\n")

        steps = [
            ("Creating API files", self.create_api_files),
            ("Building Docker image", self.build_docker),
        ]

        for step_name, step_func in steps:
            print(f"\n{'=' * 60}")
            print(f"Step: {step_name}")
            print('=' * 60)

            if not step_func():
                print(f"\n❌ Setup failed at: {step_name}")
                return False

        print("\n" + "=" * 60)
        print("✅ Setup complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Test locally: python mvp_deploy.py --test-local")
        print("2. Deploy: python mvp_deploy.py --deploy heroku")

        return True


def main():
    parser = argparse.ArgumentParser(description="Quick deployment for Fake News API")
    parser.add_argument("--export-model", action="store_true",
                        help="Export model for production")
    parser.add_argument("--test-local", action="store_true",
                        help="Test API locally")
    parser.add_argument("--build-docker", action="store_true",
                        help="Build Docker image")
    parser.add_argument("--deploy", choices=["heroku", "cloudrun"],
                        help="Deploy to cloud platform")
    parser.add_argument("--project", help="Google Cloud project ID (for cloudrun)")
    parser.add_argument("--full-setup", action="store_true",
                        help="Run complete setup")

    args = parser.parse_args()

    deploy = QuickDeployment()

    if args.export_model:
        print("⚠️  To export model, call from Python:")
        print("   from mvp_deploy import QuickDeployment")
        print("   deploy = QuickDeployment()")
        print("   deploy.export_model(pipeline_results)")

    elif args.test_local:
        deploy.test_local()

    elif args.build_docker:
        deploy.build_docker()

    elif args.deploy == "heroku":
        deploy.deploy_heroku()

    elif args.deploy == "cloudrun":
        deploy.deploy_cloud_run(args.project)

    elif args.full_setup:
        deploy.full_setup()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()