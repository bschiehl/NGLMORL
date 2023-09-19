package supervisor.normsys;

public class DefaultFact {
	Term defFact;
	String name;

	public DefaultFact(String nm, Term def) {
		defFact = def;
		name = nm;
	}
		
		
	public Term getDefault() {
		return defFact;
	}
	
	public String getName() {
		return name;
	}

}
