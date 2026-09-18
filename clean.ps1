Write-Host "Searching for __pycache__ directories..."

Get-ChildItem -Path . -Directory -Recurse -Force |
    Where-Object { $_.Name -eq "__pycache__" } |
    ForEach-Object {
        Write-Host "Removing: $($_.FullName)"
        Remove-Item -Path $_.FullName -Recurse -Force
    }

Write-Host "Done."