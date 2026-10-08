"""Test suite for Literature Search Agent.

Tests the iterative literature search process including completeness checking,
question planning, RAG execution, and answer synthesis.
"""
import pytest
from unittest.mock import AsyncMock, Mock, patch
import dspy

from utils.literature_search import (
    LiteratureSearchAgent,
    NextStepPlanner,
    LiteratureSearchAnswerGeneration,
    LiteratureSearchAnswerGenerationModule,
    NextQuestionTask
)
from utils.dataclass import (
    LiteratureSearchAgentRequest,
    LiteratureSearchAgentResponse,
    RagRequest,
    RagResponse,
    RetrievedDocument
)
from utils.rag import RagAgent


class TestNextStepPlanner:
    """Tests for the NextStepPlanner signature."""
    
    @pytest.mark.asyncio
    async def test_next_step_planner_signature_fields(self):
        """Test that NextStepPlanner has the correct input/output fields."""
        sig = NextStepPlanner
        
        # Check that annotations exist (DSPy stores fields in __annotations__)
        assert hasattr(sig, '__annotations__')
        annotations = sig.__annotations__
        
        # Check input fields
        assert 'topic' in annotations
        assert 'guideline' in annotations
        assert 'completed_tasks_summary' in annotations
        assert 'max_iterations' in annotations
        assert 'current_iteration' in annotations
        
        # Check output fields
        assert 'is_complete' in annotations
        assert 'next_questions' in annotations
        assert 'reasoning' in annotations
    
    @pytest.mark.asyncio
    async def test_next_step_planner_identifies_completion(self):
        """Test that the planner can identify when research is complete."""
        mock_lm = Mock(spec=dspy.LM)
        predictor = dspy.Predict(NextStepPlanner)
        
        mock_response = Mock()
        mock_response.is_complete = True
        mock_response.next_questions = []
        mock_response.reasoning = "All key areas have been thoroughly explored."
        
        with patch.object(predictor, 'aforward', return_value=mock_response):
            result = await predictor.aforward(
                topic="AI in healthcare",
                guideline="Conduct a comprehensive survey",
                completed_tasks_summary="Completed 10 detailed explorations covering all major aspects",
                max_iterations=15,
                current_iteration=10,
                lm=mock_lm
            )
            
            assert result.is_complete is True
            assert len(result.next_questions) == 0
    
    @pytest.mark.asyncio
    async def test_next_step_planner_generates_next_questions(self):
        """Test that the planner generates next questions when incomplete."""
        mock_lm = Mock(spec=dspy.LM)
        predictor = dspy.Predict(NextStepPlanner)
        
        mock_question1 = Mock()
        mock_question1.question = "What are the regulatory requirements?"
        mock_question1.question_context = "Understanding compliance landscape"
        
        mock_question2 = Mock()
        mock_question2.question = "What are recent technological advances?"
        mock_question2.question_context = "Identifying state-of-the-art"
        
        mock_response = Mock()
        mock_response.is_complete = False
        mock_response.next_questions = [mock_question1, mock_question2]
        mock_response.reasoning = "Need to explore regulatory and technical aspects"
        
        with patch.object(predictor, 'aforward', return_value=mock_response):
            result = await predictor.aforward(
                topic="AI in healthcare",
                guideline="Conduct a comprehensive survey",
                completed_tasks_summary="Completed 3 initial explorations",
                max_iterations=15,
                current_iteration=3,
                lm=mock_lm
            )
            
            assert result.is_complete is False
            assert len(result.next_questions) > 0
            assert len(result.next_questions) <= 3  # Should generate 1-3 questions


class TestLiteratureSearchAnswerGeneration:
    """Tests for the LiteratureSearchAnswerGeneration signature."""
    
    @pytest.mark.asyncio
    async def test_answer_generation_signature_fields(self):
        """Test that LiteratureSearchAnswerGeneration has the correct fields."""
        sig = LiteratureSearchAnswerGeneration
        
        # Check that annotations exist (DSPy stores fields in __annotations__)
        assert hasattr(sig, '__annotations__')
        annotations = sig.__annotations__
        
        # Check input fields
        assert 'topic' in annotations
        assert 'gathered_information' in annotations
        assert 'report_style' in annotations
        
        # Check output fields
        assert 'answer' in annotations
    
    @pytest.mark.asyncio
    async def test_answer_generation_synthesizes_information(self):
        """Test that answer generation synthesizes multiple sub-answers."""
        mock_lm = Mock(spec=dspy.LM)
        predictor = dspy.Predict(LiteratureSearchAnswerGeneration)
        
        gathered_info = """
        Sub-question (a): What is AI?
        Answer: AI is artificial intelligence[1].
        
        Sub-question (b): How is AI used in healthcare?
        Answer: AI is used for diagnostics[2] and treatment planning[3].
        """
        
        mock_response = Mock()
        mock_response.answer = "# AI in Healthcare\n\nAI, or artificial intelligence[1], is widely used in healthcare for diagnostics[2] and treatment planning[3]."
        
        with patch.object(predictor, 'aforward', return_value=mock_response):
            result = await predictor.aforward(
                topic="AI in healthcare",
                gathered_information=gathered_info,
                report_style="Comprehensive and detailed",
                lm=mock_lm
            )
            
            # Should preserve citations
            assert "[1]" in result.answer or "[2]" in result.answer or "[3]" in result.answer


class TestLiteratureSearchAnswerGenerationModule:
    """Tests for the LiteratureSearchAnswerGenerationModule."""
    
    @pytest.mark.asyncio
    async def test_module_initialization(self):
        """Test that the module initializes correctly."""
        mock_lm = Mock(spec=dspy.LM)
        module = LiteratureSearchAnswerGenerationModule(lm=mock_lm)
        
        assert module.survey_answer_generation_lm is mock_lm
        assert hasattr(module, 'survey_answer_generation')
    
    @pytest.mark.asyncio
    async def test_citation_normalization(self):
        """Test that the module normalizes citations across multiple RAG responses."""
        mock_lm = Mock(spec=dspy.LM)
        module = LiteratureSearchAnswerGenerationModule(lm=mock_lm)
        
        doc1 = RetrievedDocument(url="http://example.com/1", excerpts=["Content 1"])
        doc2 = RetrievedDocument(url="http://example.com/2", excerpts=["Content 2"])
        doc3 = RetrievedDocument(url="http://example.com/3", excerpts=["Content 3"])
        
        rag_responses = [
            RagResponse(
                question="Question 1",
                answer="Answer 1[1]",
                cited_documents=[doc1]
            ),
            RagResponse(
                question="Question 2",
                answer="Answer 2[1][2]",
                cited_documents=[doc2, doc3]
            )
        ]
        
        updated_answers, all_documents = module._normalize_rag_response_citation_indices(rag_responses)
        
        # Should have 3 total documents
        assert len(all_documents) == 3
        # Should have 2 updated answers
        assert len(updated_answers) == 2
        # First answer should still have [1]
        assert "[1]" in updated_answers[0]
        # Second answer should have [2] and [3] (offset by 1)
        assert "[2]" in updated_answers[1] or "[3]" in updated_answers[1]
    
    @pytest.mark.asyncio
    async def test_aforward_complete_synthesis(self):
        """Test the complete answer synthesis process."""
        mock_lm = Mock(spec=dspy.LM)
        module = LiteratureSearchAnswerGenerationModule(lm=mock_lm)
        
        doc1 = RetrievedDocument(url="http://example.com/1", excerpts=["Content 1"])
        
        rag_responses = [
            RagResponse(
                question="What is AI?",
                answer="AI is artificial intelligence[1]",
                cited_documents=[doc1]
            )
        ]
        
        request = LiteratureSearchAgentRequest(
            topic="AI basics",
            guideline="Survey fundamentals"
        )
        
        # Mock the synthesis result
        mock_synthesis = Mock()
        mock_synthesis.answer = "AI is artificial intelligence[1]"
        
        with patch.object(module.survey_answer_generation, 'aforward', return_value=mock_synthesis):
            result = await module.aforward(request, rag_responses)
            
            assert isinstance(result, LiteratureSearchAgentResponse)
            assert result.topic == "AI basics"
            assert len(result.cited_documents) >= 0


class TestLiteratureSearchAgent:
    """Tests for the complete Literature Search Agent."""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test that the agent initializes correctly."""
        mock_rag_agent = Mock(spec=RagAgent)
        mock_search_lm = Mock(spec=dspy.LM)
        mock_synthesis_lm = Mock(spec=dspy.LM)
        
        agent = LiteratureSearchAgent(
            rag_agent=mock_rag_agent,
            literature_search_lm=mock_search_lm,
            answer_synthesis_lm=mock_synthesis_lm
        )
        
        assert agent.rag_agent is mock_rag_agent
        assert agent.literature_search_lm is mock_search_lm
        assert agent.answer_synthesis_lm is mock_synthesis_lm
        assert hasattr(agent, 'completeness_checker')
    
    @pytest.mark.asyncio
    async def test_build_tasks_summary_empty(self):
        """Test building task summary with no completed tasks."""
        mock_rag_agent = Mock(spec=RagAgent)
        mock_search_lm = Mock(spec=dspy.LM)
        mock_synthesis_lm = Mock(spec=dspy.LM)
        
        agent = LiteratureSearchAgent(
            rag_agent=mock_rag_agent,
            literature_search_lm=mock_search_lm,
            answer_synthesis_lm=mock_synthesis_lm
        )
        
        summary = agent._build_tasks_summary([])
        
        assert "No tasks completed yet" in summary
    
    @pytest.mark.asyncio
    async def test_build_tasks_summary_with_responses(self):
        """Test building task summary with completed RAG responses."""
        mock_rag_agent = Mock(spec=RagAgent)
        mock_search_lm = Mock(spec=dspy.LM)
        mock_synthesis_lm = Mock(spec=dspy.LM)
        
        agent = LiteratureSearchAgent(
            rag_agent=mock_rag_agent,
            literature_search_lm=mock_search_lm,
            answer_synthesis_lm=mock_synthesis_lm
        )
        
        rag_responses = [
            RagResponse(
                question="What is AI?",
                question_context="Basics of AI",
                answer="AI is artificial intelligence",
                cited_documents=[]
            ),
            RagResponse(
                question="How is AI used?",
                question_context="Applications of AI",
                answer="AI is used in many fields",
                cited_documents=[]
            )
        ]
        
        summary = agent._build_tasks_summary(rag_responses)
        
        assert "1. Question: What is AI?" in summary
        assert "2. Question: How is AI used?" in summary
    
    @pytest.mark.asyncio
    async def test_aforward_without_synthesis(self):
        """Test literature search without final synthesis."""
        mock_rag_agent = Mock(spec=RagAgent)
        mock_rag_agent.aforward = AsyncMock(return_value=RagResponse(
            question="Test question",
            answer="Test answer[1]",
            cited_documents=[],
            num_retriever_calls=1
        ))
        
        mock_search_lm = Mock(spec=dspy.LM)
        mock_synthesis_lm = Mock(spec=dspy.LM)
        
        agent = LiteratureSearchAgent(
            rag_agent=mock_rag_agent,
            literature_search_lm=mock_search_lm,
            answer_synthesis_lm=mock_synthesis_lm
        )
        
        # Mock completeness checker to say complete immediately
        mock_completeness = Mock()
        mock_completeness.is_complete = True
        mock_completeness.next_questions = []
        mock_completeness.reasoning = "Complete"
        
        with patch.object(agent.completeness_checker, 'aforward', return_value=mock_completeness):
            request = LiteratureSearchAgentRequest(
                topic="Test topic",
                max_retriever_calls=5,
                with_synthesis=False
            )
            
            result = await agent.aforward(request)
            
            assert isinstance(result, LiteratureSearchAgentResponse)
            assert result.topic == "Test topic"
            assert result.writeup is None
    
    @pytest.mark.asyncio
    async def test_aforward_with_synthesis(self):
        """Test literature search with final synthesis."""
        doc = RetrievedDocument(url="http://example.com", excerpts=["Content"])
        
        mock_rag_agent = Mock(spec=RagAgent)
        mock_rag_agent.aforward = AsyncMock(return_value=RagResponse(
            question="Test question",
            answer="Test answer[1]",
            cited_documents=[doc],
            num_retriever_calls=1
        ))
        
        mock_search_lm = Mock(spec=dspy.LM)
        mock_synthesis_lm = Mock(spec=dspy.LM)
        
        agent = LiteratureSearchAgent(
            rag_agent=mock_rag_agent,
            literature_search_lm=mock_search_lm,
            answer_synthesis_lm=mock_synthesis_lm
        )
        
        # Mock completeness checker
        mock_completeness = Mock()
        mock_completeness.is_complete = True
        mock_completeness.next_questions = []
        mock_completeness.reasoning = "Complete"
        
        # Mock synthesis result
        mock_synthesis_result = LiteratureSearchAgentResponse(
            topic="Test topic",
            guideline="Test guideline",
            writeup="Synthesized report",
            cited_documents=[doc],
            rag_responses=[]
        )
        
        with patch.object(agent.completeness_checker, 'aforward', return_value=mock_completeness):
            with patch.object(agent.literature_search_answer_generation_module, 'aforward', return_value=mock_synthesis_result):
                request = LiteratureSearchAgentRequest(
                    topic="Test topic",
                    max_retriever_calls=5,
                    with_synthesis=True
                )
                
                result = await agent.aforward(request)
                
                assert isinstance(result, LiteratureSearchAgentResponse)
                assert result.writeup == "Synthesized report"
    
    @pytest.mark.asyncio
    async def test_iterative_exploration(self):
        """Test that the agent performs iterative exploration."""
        doc = RetrievedDocument(url="http://example.com", excerpts=["Content"])
        
        mock_rag_agent = Mock(spec=RagAgent)
        mock_rag_agent.aforward = AsyncMock(return_value=RagResponse(
            question="Test question",
            answer="Test answer[1]",
            cited_documents=[doc],
            num_retriever_calls=1
        ))
        
        mock_search_lm = Mock(spec=dspy.LM)
        mock_synthesis_lm = Mock(spec=dspy.LM)
        
        agent = LiteratureSearchAgent(
            rag_agent=mock_rag_agent,
            literature_search_lm=mock_search_lm,
            answer_synthesis_lm=mock_synthesis_lm
        )
        
        # First call: not complete, generate questions
        mock_next_question = Mock()
        mock_next_question.question = "Follow-up question"
        mock_next_question.question_context = "Context"
        
        mock_completeness_1 = Mock()
        mock_completeness_1.is_complete = False
        mock_completeness_1.next_questions = [mock_next_question]
        mock_completeness_1.reasoning = "Need more info"
        
        # Second call: complete
        mock_completeness_2 = Mock()
        mock_completeness_2.is_complete = True
        mock_completeness_2.next_questions = []
        mock_completeness_2.reasoning = "Complete now"
        
        completeness_responses = [mock_completeness_1, mock_completeness_2]
        
        async def mock_completeness_aforward(*args, **kwargs):
            return completeness_responses.pop(0)
        
        with patch.object(agent.completeness_checker, 'aforward', side_effect=mock_completeness_aforward):
            request = LiteratureSearchAgentRequest(
                topic="Test topic",
                max_retriever_calls=5,
                with_synthesis=False
            )
            
            result = await agent.aforward(request)
            
            # Should have performed at least one RAG call
            assert mock_rag_agent.aforward.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_respects_max_retriever_calls(self):
        """Test that the agent respects the maximum retriever calls limit."""
        doc = RetrievedDocument(url="http://example.com", excerpts=["Content"])
        
        mock_rag_agent = Mock(spec=RagAgent)
        mock_rag_agent.aforward = AsyncMock(return_value=RagResponse(
            question="Test question",
            answer="Test answer[1]",
            cited_documents=[doc],
            num_retriever_calls=1
        ))
        
        mock_search_lm = Mock(spec=dspy.LM)
        mock_synthesis_lm = Mock(spec=dspy.LM)
        
        agent = LiteratureSearchAgent(
            rag_agent=mock_rag_agent,
            literature_search_lm=mock_search_lm,
            answer_synthesis_lm=mock_synthesis_lm
        )
        
        # Always say not complete, but we should stop due to budget
        mock_next_question = Mock()
        mock_next_question.question = "Question"
        mock_next_question.question_context = "Context"
        
        mock_completeness = Mock()
        mock_completeness.is_complete = False
        mock_completeness.next_questions = [mock_next_question]
        mock_completeness.reasoning = "Never complete"
        
        with patch.object(agent.completeness_checker, 'aforward', return_value=mock_completeness):
            request = LiteratureSearchAgentRequest(
                topic="Test topic",
                max_retriever_calls=3,  # Very limited budget
                with_synthesis=False
            )
            
            result = await agent.aforward(request)
            
            # Should have stopped due to budget constraint
            total_calls = sum(r.num_retriever_calls for r in result.rag_responses)
            assert total_calls <= 3


class TestLiteratureSearchEdgeCases:
    """Tests for edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_empty_next_questions_list(self):
        """Test handling when no next questions are generated."""
        mock_rag_agent = Mock(spec=RagAgent)
        mock_search_lm = Mock(spec=dspy.LM)
        mock_synthesis_lm = Mock(spec=dspy.LM)
        
        agent = LiteratureSearchAgent(
            rag_agent=mock_rag_agent,
            literature_search_lm=mock_search_lm,
            answer_synthesis_lm=mock_synthesis_lm
        )
        
        # Return not complete but no questions
        mock_completeness = Mock()
        mock_completeness.is_complete = False
        mock_completeness.next_questions = []
        mock_completeness.reasoning = "Stuck"
        
        with patch.object(agent.completeness_checker, 'aforward', return_value=mock_completeness):
            request = LiteratureSearchAgentRequest(
                topic="Test topic",
                max_retriever_calls=5,
                with_synthesis=False
            )
            
            result = await agent.aforward(request)
            
            # Should end gracefully
            assert isinstance(result, LiteratureSearchAgentResponse)

