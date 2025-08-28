from dataclasses import dataclass
from typing import List
from functai import magic, step, final

# Example 1
@magic(lm="gpt-4.1")
def translator(french: str) -> str:
    """Translate French text to English."""
    english: str = final(desc="The translated English text")
    return english

translator("Bonjour le monde!")


translator("Bonjour le monde!", _prediction=True)


# Example 2
@dataclass
class Analysis:
    sentiment: str
    confidence: float
    keywords: List[str]


@magic(adapter="json", lm="gpt-4.1")
def analyze_text(text: str) -> int:
    """Analyze text for sentiment and extract key information."""

    # Step 1: Extract sentiment
    _sentiment: str = step(desc="Determine if positive, negative, or neutral")

    # Step 2: Calculate confidence
    _confidence: float = step(desc="Confidence score between 0 and 1")

    # Step 3: Extract keywords
    _keywords: List[str] = step(desc="List of important keywords")

    # Final output combines all steps
    result: Analysis = final(desc="Complete text analysis")

    return result.confidence

analyze_text("This is great!")

# Out[1]: Analysis(sentiment='The sentiment of the text is positive, as it expresses enthusiasm or approval.', confidence=0.98, keywords=['great'])


type(analyze_text)
# Out[2]: function

analyze_text.__dspy__

# Out[3]:namespace(signature=Analyze_TextSig(text -> sentiment, confidence, keywords, result__Analysis_sentiment, result__Analysis_confidence, result__Analysis_keywords
#                         instructions='Analyze text for sentiment and extract key information.'
#                         text = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Text:', 'desc': '${text}'})
#                         sentiment = Field(annotation=str required=True json_schema_extra={'desc': 'Determine if positive, negative, or neutral', '__dspy_field_type': 'output', 'prefix': 'Sentiment:'})
#                         confidence = Field(annotation=float required=True json_schema_extra={'desc': 'Confidence score between 0 and 1', '__dspy_field_type': 'output', 'prefix': 'Confidence:'})
#                         keywords = Field(annotation=List[str] required=True json_schema_extra={'desc': 'List of important keywords', '__dspy_field_type': 'output', 'prefix': 'Keywords:'})
#                         result__Analysis_sentiment = Field(annotation=str required=True json_schema_extra={'desc': 'Complete text analysis', '__dspy_field_type': 'output', 'prefix': 'Result   Analysis Sentiment:'})
#                         result__Analysis_confidence = Field(annotation=float required=True json_schema_extra={'desc': 'Complete text analysis', '__dspy_field_type': 'output', 'prefix': 'Result   Analysis Confidence:'})
#                         result__Analysis_keywords = Field(annotation=List[str] required=True json_schema_extra={'desc': 'Complete text analysis', '__dspy_field_type': 'output', 'prefix': 'Result   Analysis Keywords:'})
#                     ),
#           module=Predict(Analyze_TextSig(text -> sentiment, confidence, keywords, result__Analysis_sentiment, result__Analysis_confidence, result__Analysis_keywords
#                      instructions='Analyze text for sentiment and extract key information.'
#                      text = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Text:', 'desc': '${text}'})
#                      sentiment = Field(annotation=str required=True json_schema_extra={'desc': 'Determine if positive, negative, or neutral', '__dspy_field_type': 'output', 'prefix': 'Sentiment:'})
#                      confidence = Field(annotation=float required=True json_schema_extra={'desc': 'Confidence score between 0 and 1', '__dspy_field_type': 'output', 'prefix': 'Confidence:'})
#                      keywords = Field(annotation=List[str] required=True json_schema_extra={'desc': 'List of important keywords', '__dspy_field_type': 'output', 'prefix': 'Keywords:'})
#                      result__Analysis_sentiment = Field(annotation=str required=True json_schema_extra={'desc': 'Complete text analysis', '__dspy_field_type': 'output', 'prefix': 'Result   Analysis Sentiment:'})
#                      result__Analysis_confidence = Field(annotation=float required=True json_schema_extra={'desc': 'Complete text analysis', '__dspy_field_type': 'output', 'prefix': 'Result   Analysis Confidence:'})
#                      result__Analysis_keywords = Field(annotation=List[str] required=True json_schema_extra={'desc': 'Complete text analysis', '__dspy_field_type': 'output', 'prefix': 'Result   Analysis Keywords:'})
#                  )),
#           spec=ParsedSpec(mode='markers', inputs={'text': <class 'str'>}, outputs=[OutputDef(var_name='_sentiment', field_name='sentiment', typ=<class 'str'>, is_final=False, desc='Determine if positive, negative, or neutral', dspy_fields=['sentiment'], kind='simple', type_name=None, field_types=None), OutputDef(var_name='_confidence', field_name='confidence', typ=<class 'float'>, is_final=False, desc='Confidence score between 0 and 1', dspy_fields=['confidence'], kind='simple', type_name=None, field_types=None), OutputDef(var_name='_keywords', field_name='keywords', typ=typing.List[str], is_final=False, desc='List of important keywords', dspy_fields=['keywords'], kind='simple', type_name=None, field_types=None), OutputDef(var_name='result', field_name='result', typ=<class '__main__.Analysis'>, is_final=True, desc='Complete text analysis', dspy_fields=['result__Analysis_sentiment', 'result__Analysis_confidence', 'result__Analysis_keywords'], kind='dataclass', type_name='Analysis', field_types={'sentiment': <class 'str'>, 'confidence': <class 'float'>, 'keywords': typing.List[str]})], return_ann=<class '__main__.Analysis'>, final_var='result', doc='Analyze text for sentiment and extract key information.'),
#           adapter=<dspy.adapters.json_adapter.JSONAdapter at 0x7eca20dcb850>)
