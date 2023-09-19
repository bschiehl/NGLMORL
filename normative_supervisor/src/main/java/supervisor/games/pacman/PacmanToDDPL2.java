package supervisor.games.pacman;

import java.util.ArrayList;

import spindle.core.dom.Literal;
import spindle.core.dom.Rule;
import spindle.core.dom.RuleException;
import spindle.core.dom.RuleType;
import supervisor.games.Environment;
import supervisor.games.Game;
import supervisor.normsys.ConstitutiveNorm;
import supervisor.normsys.NormBase;
import supervisor.normsys.Term;
import supervisor.normsys.DefaultFact;

/**
 * PacmanToDDPL modified in order to accommodate parsing DDPLReasoner2. 
 * Just needed to modify the action constitutive rules method.
 * Not really necessary, just generates a smaller theory. Also needed 
 * to have a method for generating default assumptions.
 * 
 * @author emery
 *
 */

public class PacmanToDDPL2 extends PacmanToDDPL {

	public PacmanToDDPL2(NormBase nb) {
		super(nb);
	}
	
	@Override
	public void update(Environment env, ArrayList<String> actions, Game game) {
		generateActionNorms(actions);
		translateBoard((PacmanEnvironment) env);
		generateGameFacts(((PacmanGame) game).hasBlueViolated(), ((PacmanGame) game).hasOrangeViolated());
		generateRegulativeRules((PacmanEnvironment) env);
		generateConstitutiveRules((PacmanEnvironment) env);
		generateDefeaters((PacmanEnvironment) env);
		generateHierarchies();
		generateDefaultFacts();

	}
	
	
    @Override
	public void generateActionConstitutiveRules(PacmanEnvironment board) {
		try {
			ArrayList<ConstitutiveNorm> actNorms = normBase.getActionConstitutiveNorms();
			for(ConstitutiveNorm n : actNorms) {
				Rule rule = new Rule("pos:"+n.getName(), RuleType.STRICT);
				for(Term term : n.getContext()) {
					if(term.isPredicate()) {
						try {
							unaryPredicateToFact(term, board.getObject(term.getBaseObject()));
						} catch(NullPointerException e) {}
						try {
							binaryPredicateToConstitutive(term, board.getObject(term.getBaseObject()), board.getObject(term.getSateliteObject()));
						} catch(NullPointerException e) {}
						
					}
					Literal lit = termToLit(term, false);
					rule.addBodyLiteral(lit);
				}
				//positive
				for(Term term : n.getLowerTerms()) {
					Literal lit1 = termToLit(term, false);
					rule.addBodyLiteral(lit1);
				}
				Literal head1 = termToLit(n.getHigherTerm(), false);	
				rule.addHeadLiteral(head1);
				rules.add(rule);
				strategies.add(rule);
			}
		}
		catch (RuleException e) {
			e.printStackTrace();
		}
	}
    
    public void generateDefaultFacts() {
    	ArrayList<DefaultFact> defs = normBase.getDefaultFacts();
    	for(DefaultFact def : defs) {
    		try {
    			Rule rule = new Rule(def.getName(), RuleType.DEFEASIBLE);
        		Literal head = termToLit(def.getDefault(), true);
				rule.addHeadLiteral(head);
				rules.add(rule);
			} catch (RuleException e) {
				e.printStackTrace();
			}
    	}
    }
	
		



}
