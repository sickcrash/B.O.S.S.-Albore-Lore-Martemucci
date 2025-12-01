import pytest
from source_adapter.interface import VideoSourceAdapter

#Verifica che non sia possibile istanziare direttamente la classe astratta VideoSourceAdapter.
def test_cannot_instantiate_abstract_class():
    with pytest.raises(TypeError):
        adapter = VideoSourceAdapter()

#Verifica che una classe concreta che eredita da VideoSourceAdapter debba implementare tutti i metodi astratti.
def test_concrete_class_must_implement_methods():
  
    class DummyAdapter(VideoSourceAdapter):
        def open(self):
            pass

        def frames(self):
            pass

        def close(self):
            pass

    # Istanzia la classe concreta
    adapter = DummyAdapter()
    assert adapter is not None