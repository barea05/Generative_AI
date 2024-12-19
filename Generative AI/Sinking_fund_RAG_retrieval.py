from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import google.generativeai as genai
from langchain.schema import Document
from FlagEmbedding import FlagReranker
from dotenv import load_dotenv
import pdfplumber, logging
from tqdm import tqdm
import os, sys, re, json, csv
import pandas as pd
import typing_extensions as typing

class SinkingFundSchedule(typing.TypedDict):
    dueDate_Year: str
    due_Amount: float
    
class TermBond(typing.TypedDict):
    amount: float
    cuponRate: float
    maturityDate: str
    price: float
    yield_per: str
    cusip: str
    sinkingFundSchedule: typing.List[SinkingFundSchedule]

class TermBonds(typing.TypedDict):
    termBonds: typing.List[TermBond]
    

log_file_path = "/home/naveen/sinking_fund_genai/sinking_fund_linking_gemflash_main.log"

logging.basicConfig(filename=log_file_path,
                    format='%(asctime)s %(message)s',
                    filemode='a+',
                    level=logging.INFO)

load_dotenv()

path = "/FERack11_FE_documents2/EMMA_Official_Statement"
sys.path.append(path)

GROQ_API_KEY = os.getenv('GROQ_API_KEY') # GROQ_API_KEY_NK - Personal
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-70b-versatile", temperature=0) # top_p=0.9 | llama-3.1-8b-instant | llama-3.1-70b-versatile | llama3-70b-8192

# Gemini-Flash
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel("gemini-1.5-flash")

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2") # all-mpnet-base-v2
reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=False)

output_csv_file = "/home/naveen/sinking_fund_genai/output_Dec06_main.csv"

headers = [
    "Filename", "Page_Nos", "Amount", "CouponRate", "MaturityDate", "Price", 
    "Yield", "CUSIP", "SinkingFundSchedule__dueDate_Year", 
    "SinkingFundSchedule__amount", "Total_tokens"
]

with open(output_csv_file, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(headers)
    
def extract_last_processed_isin(log_file_path):
    """Extract the last processed ISIN from the log file."""
    last_isin = None
    try:
        if os.path.exists(log_file_path):
            with open(log_file_path, "r") as log_file:
                lines = log_file.readlines()
            for line in reversed(lines):
                if line:
                    match = re.search(r'([A-Z0-9-]+)\.pdf$', line)
                    if match:
                        last_isin = match.group(1)
                        break
    except Exception as e:
        logging.error(f"Error reading log file: {e}")
    return last_isin

def get_resume_index(excel_path, sheet_name, last_isin):
    """Find the index of the last processed ISIN in the Excel file."""
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    try:
        isin_column = df.columns[0]
        last_index = df[isin_column].tolist().index(last_isin)
        return last_index
    except ValueError:
        logging.warning(f"ISIN {last_isin} not found in the Excel file.")
        return 0

def extract_relevant_pages(pdf_path, keyword1, keyword2, page_limit=45):
    def find_dates(text):
        date_patterns = [
            r'\b\d{4}\b',  # 4-digit year
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # 01/01/2024 or 1-1-24
            r'\b\d{1,2} [A-Za-z]+ \d{4}\b',  # 1 January 2024
            r'\b[A-Za-z]+ \d{1,2}, \d{4}\b'  # January 1, 2024
        ]
        for pattern in date_patterns:
            if re.search(pattern, text):
                return True
        return False

    relevant_pages = set()

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = min(len(pdf.pages), page_limit)
        keyword_found = False
        
        for page_num in range(total_pages):
            if keyword_found:
                break
            
            page = pdf.pages[page_num]
            text = page.extract_text()
            
            if text:
                if any(keyword in text for keyword in keyword1):
                    lines = text.splitlines()
                    for i, line in enumerate(lines):
                        if any(keyword in line for keyword in keyword1):
                            subsequent_text = "\n".join(lines[i + 1 : i + 7])
                            if find_dates(subsequent_text):
                                relevant_pages.add(page_num + 1)
                                if page_num + 2 <= total_pages:
                                    relevant_pages.add(page_num + 2)
                                if page_num + 3 <= total_pages:
                                    relevant_pages.add(page_num + 3)
                                if page_num + 4 <= total_pages:
                                    relevant_pages.add(page_num + 4)
                                keyword_found = True
                                break

        for page_num in range(total_pages):
            page = pdf.pages[page_num]
            text = page.extract_text()

            if text and any(keyword in text for keyword in keyword2):
                relevant_pages.add(page_num + 1)

    return sorted(relevant_pages)

def prepare_retrievers(pdf_path):
    keyword1 = ["Mandatory Sinking Fund Redemption", "Mandatory Redemption", "MANDATORY REDEMPTION", "Mandatory Sinking Fund", "Sinking Fund Redemption", "Mandatory Redemption Prior to Maturity", "Scheduled Mandatory Redemption of the Series", "Mandatory Sinking Fund Redemption."]
    keyword2 = ["Term Bonds due", "Term Bond due", "Terms Bonds", "Term Bonds at", "Term Bonds,", "Term Bond", "Term Series", "TermSeries", "Term Notice", "Term Certificate"]

    relevant_pages = extract_relevant_pages(pdf_path, keyword1, keyword2)
    print("Pages To Process:", relevant_pages)

    texts = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num in relevant_pages:
            if page_num <= len(pdf.pages):
                page = pdf.pages[page_num - 1]
                text = page.extract_text()
                if text:
                    texts.append(text)

    documents = [Document(page_content=text, metadata={"source": f"Page {i+1}"}) for i, text in enumerate(texts)]
    bm25_retriever = BM25Retriever.from_documents(documents)

    vector_store = FAISS.from_documents(documents, embeddings)
    vector_retriever = vector_store.as_retriever()

    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.2, 0.8])
    return ensemble_retriever, relevant_pages

def query_retrievers(ensemble_retriever, query, top_k=7):
    results = ensemble_retriever.get_relevant_documents(query)[:top_k]
    return results

def rerank_documents(retrieved_docs, query):
    input_pairs = [[query, doc.page_content] for doc in retrieved_docs]
    scores = reranker.compute_score(input_pairs)

    for doc, score in zip(retrieved_docs, scores):
        doc.metadata["rerank_score"] = score

    reranked_docs = sorted(retrieved_docs, key=lambda x: x.metadata["rerank_score"], reverse=True)
    return reranked_docs

def query_llm_with_context(reranked_docs, query):
    context = "\n\n".join([doc.page_content for doc in reranked_docs])
    prompt_template = ChatPromptTemplate.from_template("""
    You are a financial analyst specializing in municipal bond documentation analysis. Each dates and amount extracted are very very important. Slight changes in amount or date may cause major bussiness impact for the company. You may end-up loosing your JOB.
    Using the provided context, follow the below instructions carefully:
    
    ### Context Preservation Rules:
        - Carefully scan the ENTIRE context, not just sections with obvious keywords
        - Pay special attention to tabular or table-like data
        - Extract information even if it's not in a perfect table format
        
    ### Initial Identification:
        - Search and identify only the specific lines where "Term Bonds" or "Term Bond" or "Term Series" or "Term Notice" are mentioned.
        - These terms will typically appear as a line or passage containing the relevant information such as maturity dates, amounts, and other term bond details.
        - Common formats may include but are NOT limited to, for example:
            * "[Any text] Term Bonds due [Date]"
            * "[Any text] Term Bond maturing [Date]"
            * "Term Bonds [Any text] [Date]"
            * "[Date] Term Bonds"
        - Only consider lines with these exact phrases and exclude any other lines that mention similar terminology or related but unrelated fields.
        - Search for relevant information within 3-4 lines before and after each Term Bond mention to gather all metadata related to that specific Term Bond.
        - Multiple Term Bond descriptions may be scattered across different paragraphs, and some descriptions might span multiple lines or be split by other text. Focus on extracting the metadata from the exact lines where the Term Bond is mentioned, as well as the adjacent 3-4 lines.

    ### Task Instructions:
    1. **Field-Specific Extraction Rules for Term Bonds Metdata**:
       a) Amount:
          - Only extract values that start with a "$" symbol and include commas (e.g., "$10,000,000").
          - Extract complete number including commas.
          - If multiple amounts appear, focus on the principal amount (the first amount mentioned).
       
       b) Coupon Interest Rate:
          - Must include "%" symbol.
          - Look for patterns like "5.000%" or "5%".
          - Usually appears near Term Bond description.
          - Extract exact percentage including decimals.
       
       c) Maturity Date:
          - Must include Month, Day, and Year.
          - Standard formats: "June 1, 2040", "6/1/2040", "06/01/40" and so on.
          - Match exactly with sinking fund schedules.
       
       d) Price:
          - Look for expressions like "Price:" or "priced at" followed by a percentage or plain number.
          - If no price is explicitly stated, return an empty string ('').
       
       e) Yield:
          - Search for "Yield" or "yield to maturity".
          - Must include "%" symbol.
          - If multiple yields, take the one associated with the specific term bond.
       
       f) CUSIP:
            1. First Check the Length of CUSIP:
                - If a 9-character CUSIP is present in the Term Bond line:
                    * Use the full 9-character CUSIP directly without modification.
            2. If the CUSIP is Less Than 9 Characters:
                - Check if a Base CUSIP is available in the document:
                    - Look for the first occurrence of terms such as:
                        * "CUSIP"
                        * "CUSIP No."
                        * "Base CUSIP"
                        * "Cusip*"
                        * "CUSIP†"
                    - This should be found in the header, footer, or section headers.
                - If Base CUSIP is found and only the last 3 characters are provided in the Term Bond line:
                    * Combine the extracted Base CUSIP (first 6 characters) with the last 3 characters from the Term Bond line.
                    * Example:
                        * Base CUSIP: 66328R
                        * Last 3 characters in Term Bond line: JD1
                        * Full CUSIP: 66328RJD1
                - Ensure the total length equals 9 characters (Base CUSIP + 3 characters).
            3. If Base CUSIP is not found or the CUSIP is incomplete:
                - Search for additional references to the CUSIP in the document:
                    * Tabular headers
                    * Section headers
                    * Lines immediately before the table data
            4. Validation Rule:
                - If a valid 9-character CUSIP cannot be constructed, return an empty string ('').
    
    2. Return Only Term Bond Metadata:
          - For each Term Bond instance, return only the metadata fields relevant to that specific Term Bond: Amount, Coupon Interest Rate, Maturity Date, Price, Yield, and CUSIP.
          - Ensure no extraneous data is included, and the fields are strictly limited to those identified above.
          - If any required field is missing or cannot be matched, leave it as an empty string ('').

    ### Sinking Fund Schedule Extraction:
    1. Matching Process:
        For each Term Bond maturity date identified above:
        a) **Primary Keywords:** Locate the "Mandatory Sinking Fund Redemption" or "Mandatory Redemption", or phrases like "Term Bonds Due [Maturity Date]".
        b) **Fallback Match:** If the above keywords are not present, search for the [Maturity Date] from the Term Bonds section for Mandatory Sinking Fund Redemption installment schedule.
        c) **Location Rule:** The schedule will typically be located directly below the maturity date in the document.
        d) Identify the paragraph or table containing the exact matching maturity date/year and its associated schedule.
        e) Extract Data:
            - Extract all rows and columns of data for the corresponding maturity date, ensuring completeness.
            - **Exclude any rows containing the word "Total" or rows that refer to totals or summaries**.
        f) The last row should match the maturity year and the Term Bond's redemption amount.
        g) If the maturity year does not match the last row of the table or table-like structure:
            - Check the next 2-3 lines or passages below the table to identify if additional redemption details or rows have been provided.
            - If any additional rows or details are found, include them in the extraction and validate that the last row now matches the maturity year from the term bonds data.
    
    2. Column and Row Extraction Rules:
        - Extract ALL columns and rows, ensuring that both the left side (e.g., "2019 to 2031") and right side (e.g., "2032 to 2045") are captured completely.
        - Treat the left and right columns as a single, continuous dataset:
        - Row N in the left columns corresponds to Row N in the right columns.
        - Align the extracted data into a single table:
            - Example output format:
                Year   Amortization
                2019   $60,000
                2020   $60,000
                ...
                2045   $265,000
    
    3. Handling Split Columns:
        - If the table is split into left and right parts:
            Extract the left part first (e.g., Date/Year and Amortization from "2019–2031").
            Extract the right part next (e.g., Date/Year and Amortization from "2032–2045").
        - Merge the two parts to form a continuous dataset.
    
    4. Verification of Completeness:
        Ensure that:
            - The last row includes the maturity year (e.g., 2045*) and its corresponding amortization amount.
            - The number of rows in the left and right columns matches.
        If rows are missing or misaligned, retry the extraction.
    
    5. Horizontally Split Rows (Additional Rules):
        - For rows containing repeated Date/Year and Amount pairs where the data is horizontally split (e.g., 1/15/19 $1,000,000 7/15/19 $1,030,000)
        - Extract each pair as a separate row, ensuring that the data is represented in a consistent format. Each date and its associated amount should be treated as a separate row, even when they appear in horizontal pairs.
        - Align the data into a consistent format, for example:
            Date/Year     Amount
            1/15/19     $1,000,000
            7/15/19     $1,030,000
        - If the Date and Amount appear on the same row across multiple columns:
            * Align them correctly into separate rows.
            * Ensure the extracted format mirrors the intended structure, with each Date/Amount pair appearing on its own line.
    
    **Validation**:
        - Cross-check extracted data to ensure no information is missing or misaligned.
        - If any row, column, or table data appears incomplete or incorrectly split, re-extract the data following the proper row and column format.

    ### Context:
    {context}

    ### Query:
    {query}

    IMPORTANT:
    - Return ONLY the JSON output
    - Do NOT include explanatory text
    - Do NOT skip any fields
    - Do NOT hallucinate values
    - If information is not found, use empty string ('')
    - Maintain exact format shown above
    """)
    full_prompt = prompt_template.format(context=context, query=query)
    response = model.generate_content(full_prompt, generation_config=genai.GenerationConfig(
        response_mime_type="application/json", response_schema=TermBonds, temperature=0))
    return response.text, response.usage_metadata.total_token_count
    

refinement_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a query refinement assistant. Improve the user query based on the given task instructions and make it specific and suitable for document retrieval."),
    ("user", "{instructions}"),
    ("user", "{query}"),
    ("assistant", "Refined Query:")
])

def refine_query(instructions, query):
    refined_query = llm.invoke(refinement_prompt.format(instructions=instructions, query=query))
    return refined_query.content.strip()

def process_pdfs(pdf_paths, output_json_file):
    instructions = """
    Make the query more specific for document retrieval. Focus on financial terms, schedules, and redemption details. Emphasize the metadata related to 'Term Bonds', such as maturity dates, amounts, CUSIP numbers, yield percentages, and coupon interest percentages. Additionally, ensure that references to 'Mandatory Sinking Fund Redemption' schedule or table-like structures are highlighted. Specify that the content will span only the redemption schedules, and ensure that the extraction from the term bonds of relevant data such as dates, amounts, CUSIP numbers, yield, and interest percentages is accurate and complete.
    """
    query = "What is the redemption schedule for the mandatory sinking fund and term bonds?" # Main Query!
    # query = "Extract the term bond details along with their associated mandatory sinking fund redemption schedules?"
    refined_query = refine_query(instructions, query)
    
    results = []
    
    if os.path.exists(output_json_file):
        with open(output_json_file, 'r') as json_file:
            results = json.load(json_file)
            
    csv_file_exists = os.path.exists(output_csv_file)
    
    for file_path in tqdm(pdf_paths, desc="Processing PDFs", unit="file"):
        pdf_path = os.path.join(path, f"{file_path}.pdf")
        print(f"Processing: {pdf_path}")
        logging.info(f"Started Processing for the pdf: {pdf_path}")
        try:
            ensemble_retriever, relevant_pages = prepare_retrievers(pdf_path)
        
            retrieved_docs = query_retrievers(ensemble_retriever, refined_query)
            
            reranked_docs = rerank_documents(retrieved_docs, refined_query)

            llm_response, total_tokens = query_llm_with_context(reranked_docs[:6], refined_query)
            print(f"LLM Response for {pdf_path}:\n{llm_response}\n")
            logging.info(f"Process Ends for the pdf: {pdf_path}")
            
            if llm_response.startswith("LLM Response for"):
                llm_response = llm_response.split("\n", 1)[-1]
                
            response_data = json.loads(llm_response)

            term_bonds = response_data.get("termBonds", [])
            with open(output_csv_file, mode="a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                if not csv_file_exists:
                    writer.writerow(headers)
                    csv_file_exists = True
                for bond in term_bonds:
                    for schedule in bond.get("sinkingFundSchedule", []):
                        writer.writerow([
                            file_path,
                            relevant_pages,
                            bond.get("amount", ""),
                            bond.get("couponRate", ""),
                            bond.get("maturityDate", ""),
                            bond.get("price", ""),
                            bond.get("yield_per", ""),
                            bond.get("cusip", ""),
                            schedule.get("dueDate_Year", ""),
                            schedule.get("due_Amount", ""),
                            total_tokens
                        ])
            
            results.append({"file": file_path, "pages": relevant_pages, "response": llm_response, "total_token_count": total_tokens})
            
            with open(output_json_file, 'w') as json_file:
                json.dump(results, json_file, indent=4)

        except Exception as e:
            print(f"Error processing {pdf_path}: {e} while dumping into csv or unable to read the doc")
            logging.info(f"Error processing {pdf_path}: {e} while dumping into csv or unable to read the doc")
            

def resume_processing(log_file_path, excel_path, sheet_name, output_json_file):
    """Resume processing from the point where it left off."""
    last_isin = extract_last_processed_isin(log_file_path)
    if last_isin:
        logging.info(f"Last processed ISIN: {last_isin}")
    else:
        logging.info("No ISIN found in the log file. Starting from the beginning.")
    
    start_index = get_resume_index(excel_path, sheet_name, last_isin)
    logging.info(f"Resuming from index: {start_index}")
    
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    pdf_paths = list(df['filename'])
    
    process_pdfs(pdf_paths[start_index:], output_json_file)


if __name__ == "__main__":
    # pdf_paths = ['P11569491-P11211706-P11631897','ER1070978-ER838946-ER1239805','P21717986-P21232254-P21655209','ES1331258-ES1038334-ES1441345','ER524029-ER405012-ER806598','ER875850-ER684292-ER1085969','ER622647-ER482773-ER885680','EA451857-EA350856-EA746844','ES1206232-ES942085-ES1342864','EP982845-EP762248-EP1164060','P21755075-P31125994-P31538327','EP664684-EP518076-EP919284','ER1268800-ER990623-ER1393156','RE1370108-RE1064206-RE1473972','MS274533-MS272007-MD546948','P11523932-P31109109-P31519703','EA727021-EA570224-EA966181','P11758070-P11351021-P11787307','EP1026763-EP795516-EP1197039','EP852122-EP659506-EP1061189','ER1285439-ER1002499-ER1406194','ES1075702-ES839776-ES1240801','ER562250-ER436085-ER838444','ER750744-ER583321-ER985142','EP490819-EA338949-EP779451','ER986534-ER772175-ER1173519','RE1376015-RE1068478-RE1478533','ES1241810-ES970400-ES1371375','ER752965-ER583935-ER985753','EP865459-EP670296-EP1071989','P21786545-P21371532-P21810348','EP958344-EP743346-EP1144904','P11678763-P11292066-P11722058','EP1031729-ER885190-ER1285850','EP443651-EP347312-EP744121','ES1292011-ES1010990-ES1412333','EP544488-EP425462-EP823425','ES1041291-ES813931-ES1215324','ES1313433-ES1026091-ES1428044','EA672681-EA526759-EA922968','EP305408-EP18279-EP640298','EP318695-EP27849-EP649872','MS262396-MS237704-MD463821','P11682537-P11259332-P11684802','EA284725-EA3436-EA572156','EP393880-EP309658-EP705657','EP831466-EP641722-EP1043384','RE1347154-RE1047854-RE1456255','ER857101-ER669608-ER1071437','P11401189-P11089514-P11498003','SS1384825-SS1078037-SS1485634','ER983769-ER770005-ER1171399','ER588390-ER457168-ER860130','ER983756-ER769996-ER1171389','MS268755-MS264637-MD510457','EP409815-EP315447-EP711530','MS257354-MS232662-MD453665','P11525248-P11172073-P11587852','ES1121304-ES876868-ES1278134','ER980708-ER767565-ER1168978','EA285345-EA3927-EA599473'] # Rejected Cusips
    log_file_path = "/home/naveen/sinking_fund_genai/sinking_fund_linking_gemflash_main.log"
    output_json_file = "/home/naveen/sinking_fund_genai/output_Dec06_llm.json"
    excel_path = "/home/naveen/sinking_fund_genai/5K_ISINs.xlsx"
    resume_processing(log_file_path, excel_path, sheet_name="Sheet1", output_json_file=output_json_file)
