# Reference Verification Report

## Summary
- **36 total references** in references.bib
- **36 CONFIRMED REAL** (all verified via Google Scholar, ACM DL, Springer, Wiley, arXiv, ProQuest, etc.)
- **0 FABRICATED**
- **8 metadata errors found and fixed** (author names, journal names, volume numbers, titles)

## Fixes Applied (Two Rounds)

### Round 1 (commit 9a13952)
| # | Key | Issue | Fix |
|---|-----|-------|-----|
| 1 | anderson2025growing | Type wrong | `@unpublished` → `@mastersthesis` |
| 2 | clark2025child | Author names wrong | "Dorie R" → "Daniel R.", "André B" → "Alisa B." |
| 3 | bouchaud2024auditing | Page number wrong, missing author | pages: 55→59, added Ramaciotti as 2nd author |
| 4 | habib2025auditing | Author name wrong | "Hana" → "Hussam", added Nithyanand |
| 5 | kopecky2020sharenting | Volume wrong | volume: 119→110 |

### Round 2 (commit 0870122)
| # | Key | Issue | Fix |
|---|-----|-------|-----|
| 6 | bakioglu2024digital | Author first name wrong | "Aysenur" → "Akın" |
| 7 | divon2025children | Author first name wrong, missing co-authors | "Tal" → "Tom", added Annabell & Goanta |
| 8 | keskin2023sharenting | Wrong journal, wrong author name, missing co-authors | Cureus 15(5) → Healthcare 11(10):1359; "Ahmet Demir Keskin" → full author list |
| 9 | lam2023sociotechnical | Title was article number, not actual title | "360 sociotechnical audits" → full title |
| 10 | anderson2025growing | Author initial only | "J." → "Jenna" |

## Detailed Verification Results

| # | Key | Status | Verification Source |
|---|-----|--------|---------------------|
| 1 | ribeiro2020auditing | ✅ REAL | FAccT 2020, pp.131-141, cited 924x |
| 2 | huszar2022algorithmic | ✅ REAL | PNAS 119(1), Twitter amplification study |
| 3 | haroon2023auditing | ✅ REAL | PNAS 120(50), YouTube recommendation audit |
| 4 | hussein2020measuring | ✅ REAL | PACM HCI 4(CSCW1), YouTube misinformation |
| 5 | bouchaud2024auditing | ✅ REAL | Applied Network Science 9:59, DOI 10.1007/s41109-024-00668-6 |
| 6 | habib2025auditing | ✅ REAL | arXiv:2501.15048, cited 23x |
| 7 | lam2023sociotechnical | ✅ REAL | PACM HCI 7(CSCW2), Article 360, DOI 10.1145/3610209, cited 63x |
| 8 | papadamou2020disturbed | ✅ REAL | ICWSM 14, pp.522-533, cited 212x |
| 9 | bridle2017something | ✅ REAL | Medium blog post by James Bridle, Nov 2017 |
| 10 | tahir2019bringing | ✅ REAL | ASONAM 2019, DOI 10.1145/3341161.3342913, cited 91x |
| 11 | clark2025child | ✅ REAL | J. Business Ethics, DOI 10.1007/s10551-025-05953-7 |
| 12 | bakioglu2024digital | ✅ REAL | Sociology Lens (Wiley), DOI 10.1111/johs.12456 |
| 13 | divon2025children | ✅ REAL | New Media & Society, DOI 10.1177/14614448241304657, cited 21x |
| 14 | anderson2025growing | ✅ REAL | Master's thesis, Minnesota State U Mankato, ProQuest |
| 15 | laude2024family | ✅ REAL | Jurimetrics 64(3), pp.285-308, ABA publication, cited 14x |
| 16 | abidin2015micromicrocelebrity | ✅ REAL | M/C Journal 18(5), cited 258x |
| 17 | steinberg2017sharenting | ✅ REAL | Emory Law Journal 66, p.839, cited 744x |
| 18 | kopecky2020sharenting | ✅ REAL | Children and Youth Services Review 110, cited 160x |
| 19 | keskin2023sharenting | ✅ REAL | Healthcare (MDPI) 11(10):1359, cited 75x |
| 20 | ratner2017snorkel | ✅ REAL | VLDB Endowment 11(3), foundational Snorkel paper |
| 21 | ratner2016data | ✅ REAL | NeurIPS 2016, foundational data programming paper |
| 22 | bach2019snorkel | ✅ REAL | SIGMOD 2019, Snorkel DryBell at Google |
| 23 | johnson2022survey | ✅ REAL | ACM JDIQ 14(4), DOI 10.1145/3492546, cited 62x |
| 24 | ma2023adapting | ✅ REAL | arXiv:2310.03400, cited 49x |
| 25 | gilardi2023chatgpt | ✅ REAL | PNAS 120(30), highly cited |
| 26 | tornberg2024chatgpt | ✅ REAL | arXiv:2304.06588, cited 428x; later in SSCR 2024 |
| 27 | coppa1998 | ✅ REAL | US Law, 15 U.S.C. §§ 6501-6506 |
| 28 | kosa2024 | ✅ REAL | US Bill S.1409, 118th Congress |
| 29 | illinois2024 | ✅ REAL | Illinois Child Influencer Act, 820 ILCS 152 |
| 30 | livingstone2021classifying | ✅ REAL | CO:RE report, cited 250x |
| 31 | unicef2025 | ✅ REAL | UNICEF policy brief, Dec 2025, confirmed on unicef.org |
| 32 | covington2016deep | ✅ REAL | RecSys 2016, foundational YouTube paper |
| 33 | zhou2010impact | ✅ REAL | IMC 2010, YouTube recommendation impact |
| 34 | rieder2018ranking | ✅ REAL | Convergence 24(1), YouTube search ranking |
| 35 | sandvig2014auditing | ✅ REAL | Workshop paper 2014, foundational auditing work |
| 36 | metaxa2021auditing | ✅ REAL | Foundations and Trends in HCI 14(4), comprehensive survey |
