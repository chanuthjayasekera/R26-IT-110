# GaitAI Care Auth + Verification Platform

Full-stack authentication and role dashboard starter for the AI-Based Real-Time Gait Analysis System.

## New features in this version

- Patient registration without SCA family-history questions
- Medical professional registration with:
  - medical license number
  - hospital/clinic
  - specialization
  - years of experience
  - hospital email validation
  - license proof upload
  - pending/rejected/approved verification status
- Admin login and dashboard
- Admin Dr Management approval/rejection flow
- Admin Patient Management
- Medical professionals cannot access their dashboard until admin approval
- Patient header with:
  - Normal vs Abnormal Detection
  - SCA and KOA Detection
  - PD Detection
  - Rehab & Recommendation
- Medical professional header with:
  - Upload Risks
  - Upload Recommendations
  - Upload Rehabilitation
  - Engage Patient Profiles
- Admin header with:
  - Patient Management
  - Dr Management
- Profile avatar initials in header
- Profile page with update details and profile picture upload
- Light blue/white professional footer inspired by your sample
- Gait-related images and system-relevant quotes

## Default admin account

Email:

```txt
admin@gaitai.local
```

Password:

```txt
Admin@123!
```

## Run backend

```powershell
cd server
npm install
npm run dev
```

Backend:

```txt
http://localhost:5000
```

## Run frontend

Open a second terminal:

```powershell
cd client
npm install
npm run dev
```

Frontend:

```txt
http://localhost:5173
```

## Important if you already ran the old version

If you already created an old SQLite database, delete it before running this updated version because the admin role and verification columns are new.

PowerShell:

```powershell
cd server
Remove-Item -Recurse -Force data
mkdir data
npm run dev
```

## Demo flow

1. Start backend.
2. Start frontend.
3. Register a patient.
4. Patient should directly access the patient dashboard.
5. Register a medical professional with license proof and hospital email.
6. Medical professional should see pending approval.
7. Login as admin.
8. Go to Dr Management.
9. Approve the medical professional.
10. Login as that medical professional again.
11. Medical professional dashboard should open.

## Notes

SQLite is used for university prototype and local development. For real production deployment, migrate to PostgreSQL or MySQL.
