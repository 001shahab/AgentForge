AgentForge Documentation
========================

Welcome to AgentForge, a lightweight framework for building modular AI agents.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   skills
   integrations
   api


Installation
------------

Install AgentForge using pip::

    pip install agentforge[openai]


Quick Start
-----------

Create your first agent::

    from agentforge import Agent, WebScraperSkill, ContentGenerationSkill
    from agentforge.integrations import OpenAIBackend

    agent = Agent(
        skills=[WebScraperSkill(), ContentGenerationSkill()],
        llm=OpenAIBackend()
    )
    
    result = agent.run({"url": "https://example.com"})
    print(result["generated"])


API Reference
-------------

.. automodule:: agentforge
   :members:

.. automodule:: agentforge.core
   :members:

.. automodule:: agentforge.skills
   :members:

.. automodule:: agentforge.integrations
   :members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


About
-----

AgentForge is designed and developed by Prof. Shahab Anbarjafari from 
3S Holding OÜ, Tartu, Estonia.

Contact: shb@3sholding.com

