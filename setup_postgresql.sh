#!/bin/bash
# PostgreSQL Setup Script for Domain Sticks
# Run this script with sudo privileges

echo "🐘 Setting up PostgreSQL for Domain Sticks..."

# Check if PostgreSQL is running
if ! systemctl is-active --quiet postgresql; then
    echo "📍 Step 1: Initialize PostgreSQL database cluster"
    sudo -u postgres initdb --locale=C.UTF-8 --encoding=UTF8 -D '/var/lib/postgres/data'
    
    echo "📍 Step 2: Enable and start PostgreSQL service"
    sudo systemctl enable postgresql
    sudo systemctl start postgresql
    
    # Wait for PostgreSQL to start
    sleep 3
else
    echo "✅ PostgreSQL is already running"
fi

echo "📍 Step 3: Create database and user"
sudo -u postgres psql << EOF
-- Create database
CREATE DATABASE domain_sticks;

-- Create user
CREATE USER domain_user WITH PASSWORD 'domain_sticks_2024';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE domain_sticks TO domain_user;

-- Grant schema privileges
\c domain_sticks
GRANT ALL ON SCHEMA public TO domain_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO domain_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO domain_user;

-- Show created database and user
\l
\du

EOF

echo "✅ PostgreSQL setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Update your .env file with:"
echo "   DATABASE_URL=postgresql://domain_user:domain_sticks_2024@localhost:5432/domain_sticks"
echo ""
echo "2. Test the connection:"
echo "   source .venv/bin/activate"
echo "   python -c \"from src.models.database import DatabaseManager; db = DatabaseManager(); db.create_tables(); print('✅ PostgreSQL connection successful')\""
