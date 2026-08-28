Write-Host "Setting Google Cloud project..."
gcloud config set project nyc-mobility-showcase

Write-Host "Deploying NYC Mobility Showcase to Cloud Run..."
gcloud run deploy nyc-mobility-showcase `
  --source . `
  --region us-east1 `
  --allow-unauthenticated `
  --memory 2Gi `
  --cpu 1 `
  --min 0 `
  --max 3 `
  --service-account="nyc-mobility-runner@nyc-mobility-showcase.iam.gserviceaccount.com" `
  --add-volume="name=mobility-data,type=cloud-storage,bucket=nyc-mobility-showcase-data-jesingson,readonly=true" `
  --add-volume-mount="volume=mobility-data,mount-path=/app/data/processed/1.3.1.final_tables"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Cloud Run deployment failed."
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Deployment complete."
Write-Host "App URL:"
Write-Host "https://nyc-mobility-showcase-148034634843.us-east1.run.app"