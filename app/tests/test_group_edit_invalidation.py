"""The API tells the UI whether the accepted group edit invalidated output."""
import copy
import json
from pathlib import Path
import pytest
from automix import server as srvmod
from test_capture_groups import srv, _new_session, _upload, _wav_bytes, _post_json

@pytest.fixture()
def group_session(srv):
    base, _ = srv
    sid = _new_session(base)
    for name in ('mic_a.wav','mic_b.wav','mic_c.wav'):
        assert _upload(base,sid,name,_wav_bytes(220,dur=.1))[0]==200
    return base,sid,srvmod._sessions[sid]

def post(fixture,groups):
    base,sid,_=fixture
    code,body,_=_post_json(base,f'/api/session/{sid}/capture_groups',{'groups':groups})
    return code,json.loads(body)

def valid_output(session):
    path=Path(session.dir)/'mix.wav';path.write_bytes(b'previous output sentinel')
    session.mix_path=str(path)
    return path

def test_same_declaration_preserves_the_current_output(group_session):
    group=[{'id':'room','members':['mic_a','mic_b'],'pan':0}]
    code,body=post(group_session,group)
    assert code==200 and body['changed'] is True
    session=group_session[2];path=valid_output(session)
    code,body=post(group_session,group)
    assert code==200 and body['changed'] is False
    assert session.mix_path==str(path) and path.read_bytes()==b'previous output sentinel'
    # Omission of the default pan is equivalent; member ordering is retained.
    code,body=post(group_session,[{'id':'room','members':['mic_a','mic_b']}])
    assert code==200 and body['changed'] is False
    assert session.mix_path==str(path)

@pytest.mark.parametrize('edited',[
    [{'id':'renamed','members':['mic_a','mic_b'],'pan':0}],
    [{'id':'room','members':['mic_a','mic_c'],'pan':0}],
    [{'id':'room','members':['mic_a','mic_b'],'pan':.2}],
    [],
    [{'id':'room','members':['mic_a','mic_b'],'mode':'stereo_pair','pan':0,'trim_db':0}],
])
def test_effective_changes_invalidate_but_keep_output_bytes(group_session,edited):
    assert post(group_session,[{'id':'room','members':['mic_a','mic_b'],'pan':0}])[0]==200
    session=group_session[2];path=valid_output(session)
    code,body=post(group_session,edited)
    assert code==200 and body['changed'] is True
    assert session.mix_path is None
    assert path.read_bytes()==b'previous output sentinel'

def test_rejected_edit_preserves_declaration_and_output(group_session):
    original=[{'id':'room','members':['mic_a','mic_b'],'pan':0}]
    assert post(group_session,original)[0]==200
    session=group_session[2];path=valid_output(session);before=copy.deepcopy(session.capture_groups)
    code,body=post(group_session,[{'id':'room','members':['mic_a','mic_b'],'pan':3}])
    assert code==400 and 'changed' not in body
    assert session.capture_groups==before and session.mix_path==str(path)
    assert path.read_bytes()==b'previous output sentinel'

def test_stereo_pair_order_and_shared_gain_are_effective_changes(group_session):
    group=[{'id':'pair','members':['mic_a','mic_b'],'mode':'stereo_pair','pan':0,'trim_db':0}]
    assert post(group_session,group)[1]['changed'] is True
    assert post(group_session,group)[1]['changed'] is False
    reverse=copy.deepcopy(group);reverse[0]['members'].reverse()
    assert post(group_session,reverse)[1]['changed'] is True
    gain=copy.deepcopy(reverse);gain[0]['trim_db']=2
    assert post(group_session,gain)[1]['changed'] is True
