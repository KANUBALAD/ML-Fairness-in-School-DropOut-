# MOOC datasets referenced by Jha, Ghergulescu & Moldovan (CSEDU 2019)

Paper: "OULAD MOOC Dropout and Result Prediction using Ensemble, Deep
Learning and Regression Techniques" — DOI 10.5220/0007767901540164
(https://www.scitepress.org/Papers/2019/77679/77679.pdf, copy saved as
`paper_Jha2019_CSEDU.pdf`).

The paper's Table 1 surveys the following MOOC datasets. Status of each:

## 1. OULAD — Open University Learning Analytics Dataset  [DOWNLOADED]
Folder: `OULAD/`
- Official source: https://analyse.kmi.open.ac.uk/open_dataset
  (licence CC BY 4.0; Kuzilek, Hlosta & Zdrahal 2017)
- The official CDN link (`schools.stem.open.ac.uk/cdn/files/anonymisedData.zip`)
  is currently dead (404), so the file was retrieved from the Internet
  Archive copy of the official `analyse.kmi.open.ac.uk/open-dataset/download`
  endpoint.
- 7 official CSV files, all present:
  - `studentInfo.csv`      — 32,593 students; demographics: **gender**, region,
    highest_education, imd_band, age_band, disability; `final_result`
    (Pass / Withdrawn / Fail / Distinction)
  - `studentRegistration.csv` — enrolments per module-presentation
  - `studentAssessment.csv`   — assessment submissions/scores
  - `studentVle.csv`          — 10.6M VLE clickstream rows
  - `assessments.csv`, `vle.csv`, `courses.csv` — module metadata
- Useful for our fairness work: gender is a sensitive attribute
  (54.8% M / 45.2% F); dropout (Withdrawn) ≈ 31.2% with a small gender gap
  (F 30.5% vs M 31.7%).

## 2. KDD Cup 2015 (XuetangX MOOC dropout)  [DOWNLOADED]
Folder: `KDD_Cup_2015/`
- Competition data: 39 courses, 7 CSV files (as cited in the paper's Table 1).
- Official competition page (biendata.xyz) requires an account; the original
  moocdata.cn host is defunct. Downloaded from the public GitHub mirror
  `noonlay/MOOCdata` (file `KDDCUP2015data.7z`, Git-LFS object, sha256
  dba6afe9...30688395 bytes).
- Contents (train/test split):
  - `train/enrollment_train.csv` (72,395 learner-course records),
    `test/enrollment_test.csv` (24,013)
  - `train/log_train.csv` (4.88M behaviour logs), `test/log_test.csv` (1.62M)
  - `train/truth_train.csv`, `test/truth_test.csv` (dropout labels)
  - `train/object.csv`, `train/date.csv` (course structure/dates;
    duplicated in test/)
- NOTE: contains **no demographic columns** (the paper itself notes KDD Cup /
  XuetangX lack student demographics), so it is useful for label-imbalance
  analysis but has no sensitive attribute.

## 3. HarvardX-MITx Person-Course (2013)  [MANUAL STEP NEEDED]
Folder: `HarvardX_MITx/` (currently empty)
- Harvard Dataverse: doi:10.7910/DVN/26147
  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/26147
- File: `HXPC13_DI_v3_11-13-2019.tab` (39.4 MB), plus de-identification and
  documentation PDFs. Contains gender, birth year, education level, etc.
- Download requires accepting a Guestbook (terms of use) — please click the
  "Download" button on the page above (no scripted download).

## 4. XuetangX (Hong et al. 2017 / Qiu et al. 2016)  [NOT DOWNLOADABLE]
- Hosted on moocdata.cn (now defunct/ICP-parked); files were
  registration-gated, no public mirror found. If needed, request from the
  authors (Qiu et al. 2016, "Modeling and predicting learning outcomes in
  MOOCs").

## 5. Khan Academy (Piech et al. 2015)  [NOT DOWNLOADABLE]
- 1.4M exercise attempts, 47,495 students, 69 exercise types (Deep Knowledge
  Tracing paper). No public mirror of the raw data; the companion repo
  (github.com/chrispiech/DeepKnowledgeTracing) only ships ASSISTments and
  synthetic subsets. Request from the authors if needed.

---

Notes:
- OULAD and KDD Cup 2015 are the two immediately usable additions.
- All files were obtained from public/archived official sources; see the
  licence terms of each dataset before reuse.
