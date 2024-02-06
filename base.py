from typing import Any, Dict, List, Optional
from llama_index.llama_pack.base import BaseLlamaPack
from llama_index.readers import PDFReader
from llama_index import ServiceContext, SimpleDirectoryReader
from llama_index.schema import NodeWithScore
from llama_index.response_synthesizers import TreeSummarize
from llama_index.llms import OpenAI
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from pydantic import BaseModel, Field

try:
    from llama_index.llms.llm import LLM
except ImportError:
    from llama_index.llms.base import LLM

model_path = 'D:\Models\openhermes-2.5-mistral-7b.Q4_K_M.gguf'

llm = LlamaCPP(
    model_path=model_path,
    temperature=0.1,
    max_new_tokens=4500,
    context_window=8000,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": 1},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

QUERY_TEMPLATE = """
You are an expert fraud detector. 
You job is to decide if the case should be investigated further for fraud based on the uploaded report and a list of potential fraud criteria. You may query your existing knowledge for additional information as needed, such as average procedure length or cost to identify outliers:

### Fraud Criteria
{fraud_criteria}

### Uploaded Report
{uploaded_report}
"""


class CriteriaDecision(BaseModel):
    """The decision made based on a single criteria"""

    criterion: str = Field(description="The criterion being evaluated")
    decision: bool = Field(description="The decision made based on the criterion")
    reasoning: str = Field(description="The reasoning behind the decision")

    
class FraudCriteria(BaseModel):
    """The elements used to detect fraud"""

    FRAUD_CRITERIA = [
        "Intentionally billing for unnecessary medical services or items.",
        "Intentionally billing for services or items not provided.",
        "Billing for multiple codes for a group of procedures that are covered in a single global billing code.",
        "Billing for services at a higher level of complexity than provided.",
        "Knowingly treating and claiming reimbursement for someone other than the eligible beneficiary.",
        "Knowingly collaborating with beneficiaries to file false claims for reimbursement.",
        "Writing unnecessary prescriptions, or altering prescriptions, to obtain drugs for personal use or to sell them.",
        "Offering, soliciting, or paying for beneficiary referrals for medical services or items.",
        "Knowingly billing for an ineligible beneficiary.",
        "Uses an improper code to bill for a higher priced service or product when a lower priced service or product was provided.",
        "Bills Medicaid for non-covered services by describing them as covered services.",
        "Misrepresents a patients diagnosis or condition and/or bills  for a service or product that is not medically necessary",
        "Falsifies medical records or supporting documentation",
        "Receives or gives a kickback (money or some other thing of value) for referral of a  patient for medical services or products",
        "Falsifies a physicians certificate of medical necessity.",
        "Bills Medicaid for drugs dispensed without a lawful prescription.",
        "Creates false employee time sheets and bills  for services not rendered.",
    ]
class FraudScreenerDecision(BaseModel):
   

    """The decision made by the fraud screener"""

    criteria_decisions: List[CriteriaDecision] = Field(
        description="The decisions made based on the criteria"
    )
    overall_reasoning: str = Field(
        description="The reasoning behind the overall decision"
    )
    overall_decision: bool = Field(
        description="The overall decision made based on the criteria"
    )


def _format_criteria_str(criteria: List[str]) -> str:
    criteria_str = ""
    for criterion in criteria:
        criteria_str += f"- {criterion}\n"
    return criteria_str


class FraudScreener(BaseLlamaPack):
    def __init__(
        self, fraud_criteria: List[str], uploaded_report: str, llm: Optional[LLM] = None, mime_type: Optional[str] = None
    ) -> None:
        if mime_type is None:
            raise ValueError("mime_type must be provided")
        self.uploaded_report = uploaded_report
       
        if mime_type == 'application/pdf':
            self.reader = PDFReader()
      
        elif mime_type == 'text/plain':
            self.reader = SimpleDirectoryReader(input_files=[uploaded_report])
            uploaded_report = self.reader.load_data()
        else:
            raise ValueError(f"Unsupported MIME type: {mime_type}")
        llm = llm or OpenAI(model = "gpt-4")
        service_context = ServiceContext.from_defaults(llm=llm)
        self.synthesizer = TreeSummarize(
            output_cls=FraudScreenerDecision, service_context=service_context
        )
        criteria_str = _format_criteria_str(fraud_criteria)
        self.query = QUERY_TEMPLATE.format(
            fraud_criteria=fraud_criteria, uploaded_report=uploaded_report
        )
       
    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {"reader": self.reader, "synthesizer": self.synthesizer}


    def run(self, uploaded_report: str) -> Any:

        docs = self.reader.load_data(uploaded_report)

    
        output = self.synthesizer.synthesize(
            query = self.query,
            nodes = [NodeWithScore(node=doc, score=1.0) for doc in docs],
        )
       

        return output.response
        
   