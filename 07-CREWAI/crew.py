from crewai import Crew, Agent, Task, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

load_dotenv()

search_tool = SerperDevTool(n=2)

topic = "Brainrot because of short form content"

llm = LLM(
    model='gpt-4o-mini',  # Changed to a more stable model
    temperature=0.2
)

senior_research_agent = Agent(
    role='Senior Research Agent',
    goal=f"Research, analyze and report on this topic: {topic}, you can use the search tool to gather information.",
    tools=[search_tool],
    llm=llm,
    allow_delegation=True,
    verbose=True,
    backstory="""
    You are a highly experienced research agent with a deep understanding of various fields. Your expertise allows you to tackle complex research tasks and provide detailed insights. You are skilled in using advanced tools and methodologies to gather and analyze information effectively. Your goal is to conduct in-depth research and deliver comprehensive reports on a wide range of topics, ensuring accuracy and depth in your findings.
    You are also capable of delegating tasks to other agents when necessary, ensuring that the research process is efficient and thorough.
    """
)

content_writer_agent = Agent(
    role='Content Writer Agent',
    goal='Create engaging and informative content based on research findings.',
    llm=llm,
    allow_delegation=False,
    verbose=True,
    backstory="""
    You are a skilled content writer with a knack for transforming complex research findings into engaging and informative articles. Your writing is clear, concise, and tailored to the target audience. You excel at synthesizing information and presenting it in a way that is both accessible and compelling. Your goal is to create high-quality content that effectively communicates research insights and engages readers.
    You focus on delivering well-structured articles that are not only informative but also enjoyable to read. You do not delegate tasks, ensuring that your content is crafted with your unique voice and style.
    """
)

research_task = Task(
    description=f"""Conduct research on the topic: {topic}. Use the search tool to gather information and provide a detailed report.
    Ensure that the report is comprehensive and well-structured.
    The report should include key findings, insights, and any relevant data or statistics.
    Don't aim for quantity, aim for quality and depth in your research.
    Keep things to the point and avoid unnecessary fluff.
    """,
    expected_output='A detailed yet to the point report on the topic, including key findings and insights.',
    agent=senior_research_agent,  # Changed from 'agents' to 'agent' - single agent assignment
)

crew = Crew(
    tasks=[research_task],
    agents=[senior_research_agent, content_writer_agent],
)

if __name__ == "__main__":
    result = crew.kickoff()
    print("\n" + "="*50)
    print("FINAL RESULT:")
    print("="*50)
    print(result)