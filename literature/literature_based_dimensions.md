# Literature-Based Exploitation Dimensions for Kidfluencer Content

## Key Sources and Their Coding Schemes

### 1. Clark & Jno-Charles (2025) - "Five Fundamental Threats" Framework
- Journal of Business Ethics, cited 25 times
- Based on UNCRC (UN Convention on the Rights of the Child)
- **Five Threats:**
  1. Right to Consent (Art. 12) - child's inability to consent to content creation
  2. Right to Privacy (Art. 16) - exposure of private moments
  3. Right to Economic Protection (Art. 32) - labor without fair compensation
  4. Right to Freedom from Harm (Art. 19) - physical/psychological harm
  5. Right to Freedom of Expression (Art. 13) - scripted vs authentic expression

### 2. Papadamou et al. (2020) - "Disturbed YouTube for Kids" (ICWSM, cited 211)
- Content categories: suitable, disturbing, restricted, irrelevant
- Features used: title, tags, description, thumbnail visual features, audio features
- **Key insight**: Used multimodal features (text + visual + audio) for classification
- Built classifier with 84.3% accuracy

### 3. Divon et al. (2025) - "Children as Concealed Commodities"
- Ethnographic content analysis
- Key concepts: "concealed commodification" - children appear to be playing but are actually working
- Distinction between "authentic play" vs "instrumentalized play"

### 4. Freitas (2024) - "Between Play and Exploitation"
- Concept of "playbour" - play that is instrumentalized as labor
- Children's agency vs parental control in content creation

### 5. Hudders & Beuckels (2024) - "Kidfluencing as New Form of Child Labor"
- Journal of Children and Media
- Focus on commercial exploitation strategies embedded in content
- Whether children feel commercially exploited

### 6. Archer & Delmo (2025) - "Kidfluencers, PR and Human Rights"
- Uses UNCRC as framework
- Children as "brand extensions" of parents (Abidin, 2015)
- PR practitioners working with/through children

## Synthesis: Literature-Supported Dimensions

Based on the above, our dimensions should map to established frameworks:

| Our Dimension | UNCRC Article | Clark 2025 Threat | Other Support |
|--------------|---------------|-------------------|---------------|
| **Performative labor** | Art. 32 (Economic protection) | Economic exploitation | Freitas 2024 "playbour", Divon 2025 "concealed commodification" |
| **Emotional bait** | Art. 19 (Freedom from harm) | Freedom from harm | Clark 2025 "manufactured emotional scenarios" |
| **Narrative conflict** | Art. 19 (Freedom from harm) | Freedom from harm | Clark 2025 "scripted conflict", Divon 2025 |
| **Challenge format** | Art. 32 (Economic protection) | Economic exploitation | ILO Conv. 182 (hazardous work conditions) |
| **Commercial content** | Art. 13 (Freedom of expression) | Freedom of expression | Hudders 2024, Archer 2025 "brand extensions" |

## Dimensions We're MISSING (from literature):

1. **Privacy violation** (Art. 16) - sharing embarrassing/private moments
   - Examples: potty training videos, tantrums, medical procedures
   - Clark 2025: "exposure of intimate family moments"
   
2. **Consent indicators** (Art. 12) - signs child didn't choose to participate
   - Examples: child visibly reluctant, crying, being directed by parent
   - Only detectable via CV (not title alone)

3. **Age-inappropriate content** - content unsuitable for the child's age
   - Examples: dating themes for young children, scary pranks on toddlers
   - Papadamou 2020: "disturbing content targeting young children"

## Rule-Based LF Justification:

| Rule LF | Justification |
|---------|---------------|
| ALL CAPS ratio | Clickbait literature (Potthast et al. 2016, Chen et al. 2015) - sensationalist formatting correlates with manufactured content |
| Exclamation marks | Same clickbait literature |
| Conflict keywords | Clark 2025 "manufactured conflict scenarios" |
| Challenge keywords | ILO definition of child labor: "work that deprives children of their childhood" |
| Prank keywords | Clark 2025 "psychological harm through pranks" |
| Emotional keywords | UNCRC Art. 19 "protection from all forms of violence" |
| Organic keywords | Negative indicator - Divon 2025 "authentic family activities" |
| Roleplay keywords | Freitas 2024 "playbour" - play instrumentalized as labor |
| Question clickbait | Clickbait literature - curiosity gap exploitation |
| Urgency words | Clickbait literature - sensationalism |

## Recommended Changes to Our Framework:

1. **Keep all 5 LLM dimensions** - all have strong literature support
2. **Add "privacy_violation"** as 6th dimension - very well supported by UNCRC Art. 16
3. **Rule-based LFs are justified** by clickbait detection literature as PROXIES for exploitation
   - Key framing: "We use clickbait indicators as observable proxies for content that prioritizes engagement over child welfare"
   - Citation: Potthast et al. (2016) "A Stylometric Inquiry into Hyperpartisan and Fake News"
4. **CV-based LFs** are justified by Papadamou 2020's multimodal approach
