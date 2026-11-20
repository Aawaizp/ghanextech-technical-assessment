import unittest
from unittest.mock import patch

from agent.core import GroqToolAgent
from agent.registry import execute_tool
from test_agent import model_reply, tool_call


class LoopSafetyTests(unittest.TestCase):
    def test_completion_check_executes_missing_calculations(self):
        replies = [
            model_reply(tool_calls=[tool_call('weather', 'get_weather', '{"location":"Solapur"}')]),
            model_reply('The temperature is 21.5 and twice it is 43.'),
            model_reply(tool_calls=[
                tool_call('double', 'calculate', '{"expression":"21.5*2"}'),
                tool_call('zero', 'calculate', '{"expression":"1/0"}'),
            ]),
            model_reply('Twice is 43; division by zero is undefined.'),
        ]
        with GroqToolAgent(['test']) as agent:
            with patch.object(agent, '_ask_model', side_effect=replies), patch('agent.core.execute_tool', side_effect=[
                {'ok': True, 'data': {'temperature_c': 21.5}},
                {'ok': True, 'data': {'result': 43}},
                {'ok': False, 'error': 'Division by zero'},
            ]) as execute:
                result = agent.run([{'role': 'user', 'content': 'Get Solapur temperature, double it and calculate 1/0'}])
                self.assertEqual(execute.call_count, 3)
                self.assertEqual([entry['name'] for entry in result.tool_trace], ['get_weather', 'calculate', 'calculate'])
                self.assertFalse(result.tool_trace[-1]['result']['ok'])

    def test_concept_question_does_not_force_weather(self):
        with GroqToolAgent(['test']) as agent:
            with patch.object(agent, '_ask_model', return_value=model_reply('Moving air.')) as ask:
                agent.run([{'role': 'user', 'content': 'What causes wind?'}])
                self.assertEqual(ask.call_args.kwargs['tool_choice'], 'auto')

    def test_repeated_call_is_not_executed_twice_and_context_survives(self):
        replies = [model_reply(tool_calls=[tool_call(str(i), 'calculate', '{"expression":"2*4"}')]) for i in range(2)]
        replies.append(model_reply('It is 8.'))
        with GroqToolAgent(['test']) as agent:
            with patch.object(agent, '_ask_model', side_effect=replies) as ask, patch('agent.core.execute_tool', return_value={'ok': True, 'data': {'result': 8}}) as execute:
                result = agent.run([{'role': 'user', 'content': 'Remember Solapur'}, {'role': 'assistant', 'content': 'Okay'}, {'role': 'user', 'content': 'Calculate 2*4'}])
                self.assertEqual(execute.call_count, 1)
                self.assertTrue(result.tool_trace[1]['reused'])
                self.assertIn('Solapur', ask.call_args.args[0][1]['content'])

    def test_failure_preserves_trace(self):
        with GroqToolAgent(['test'], max_tool_rounds=1) as agent:
            with patch.object(agent, '_ask_model', side_effect=[model_reply(tool_calls=[tool_call('a', 'calculate', '{"expression":"2+2"}')]), RuntimeError('Connection failed')]):
                result = agent.run([{'role': 'user', 'content': '2+2'}])
                self.assertTrue(result.is_error)
                self.assertEqual(result.tool_trace[0]['result']['data']['result'], 4)

    def test_registry_rejects_extra_arguments_and_nonfinite_amount(self):
        self.assertFalse(execute_tool('get_weather', {'location': 'Solapur', 'forecast_days': 1})['ok'])
        for value in (True, float('nan'), float('inf'), -1):
            self.assertFalse(execute_tool('convert_currency', {'amount': value, 'from_currency': 'USD', 'to_currency': 'INR'})['ok'])
